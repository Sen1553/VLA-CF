import collections
import dataclasses
import logging
import math
import pathlib
import json
from typing import Dict, List, Literal, Optional
import os

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
try:
    # Optional import for segmentation-enabled env
    from libero.libero.envs import SegmentationRenderEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    SegmentationRenderEnv = None  # type: ignore
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from datetime import datetime

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
InputModality = Literal["language", "vision", "vision_language"]


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    hide_grasped_object: bool = True
    starting_task_id: int = 0
    # Validate server-side masking; observations and prompts are not changed here.
    expected_modality: Optional[InputModality] = None

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_cf_long"  # Also supports libero_cf_probe (alias: libero-cf-probe).
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    use_custom_conditions: bool = True


def _resolve_input_modality(
    server_metadata: Dict, expected_modality: Optional[InputModality]
) -> Optional[InputModality]:
    """Only label a run when the server's modality and mask flags agree."""
    masks_by_modality = {
        "language": {"mask_language_tokens": False, "mask_vision_tokens": True},
        "vision": {"mask_language_tokens": True, "mask_vision_tokens": False},
        "vision_language": {"mask_language_tokens": False, "mask_vision_tokens": False},
    }
    if expected_modality is not None and expected_modality not in masks_by_modality:
        raise ValueError(f"Unsupported expected modality: {expected_modality!r}")

    metadata = server_metadata if isinstance(server_metadata, dict) else {}
    modality = metadata.get("input_modality")
    errors = []
    if not isinstance(modality, str) or modality not in masks_by_modality:
        errors.append(f"input_modality must be one of {tuple(masks_by_modality)}, got {modality!r}")
    else:
        for key, expected_mask in masks_by_modality[modality].items():
            actual_mask = metadata.get(key)
            if type(actual_mask) is not bool or actual_mask != expected_mask:
                errors.append(f"{key} must be {expected_mask!r} for {modality!r}, got {actual_mask!r}")

    if metadata.get("model_backend") != "jax":
        errors.append(f"model_backend must be 'jax', got {metadata.get('model_backend')!r}")
    if expected_modality is not None:
        if modality != expected_modality:
            errors.append(f"expected input_modality={expected_modality!r}, got {modality!r}")
        if errors:
            raise ValueError("Server modality validation failed: " + "; ".join(errors))
    elif errors:
        logging.warning("Server modality is unverified; recording input_modality=null: %s", "; ".join(errors))
        return None
    return modality


def _json_metadata_default(value):
    """Preserve NumPy metadata received through msgpack as JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    # Resolve aliases before looking up condition files and naming rollout directories.
    args = dataclasses.replace(args, task_suite_name=task_suite.name)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")

    # Validate the handshake before creating outputs or launching simulator rollouts.
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    input_modality = _resolve_input_modality(server_metadata, args.expected_modality)

    global exp_name
    modality_suffix = f"-{input_modality}" if input_modality is not None else ""
    exp_name = f"{args.task_suite_name}{modality_suffix}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    save_path = f"rollouts/{exp_name}"
    video_out_path = f"{save_path}/videos"
    eef_pointcloud_out_path = f"{save_path}/eef_pointclouds"
    trajectory_out_path = f"{save_path}/trajectories"

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(eef_pointcloud_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trajectory_out_path).mkdir(parents=True, exist_ok=True)

    # Setup logging to both console and file
    _setup_logging(video_out_path, args.task_suite_name)
    with open(pathlib.Path(save_path) / "run_config.json", "w", encoding="utf-8") as fjson:
        json.dump(
            {
                "args": dataclasses.asdict(args),
                "server_metadata": server_metadata,
                "input_modality": input_modality,
            },
            fjson,
            indent=2,
            default=_json_metadata_default,
        )
    logging.info("Server input modality: %s", input_modality if input_modality is not None else "unknown")

    # Load per-task custom conditions if provided
    conditions_db = _load_custom_conditions_db(args.task_suite_name) if args.use_custom_conditions else None
    if conditions_db is not None:
        logging.info(f"Loading custom conditions from {args.task_suite_name}")
    else:
        logging.info(f"No custom conditions provided")

    task_step_limit = getattr(task_suite, "get_task_max_steps", None)
    if task_step_limit is not None:
        max_steps = None  # Mixed suites set their source-specific limit inside the task loop.
    elif args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    elif args.task_suite_name == "libero_cf_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_cf_spatial_focused":
        max_steps = 220
    elif args.task_suite_name == "libero_cf_ood":
        max_steps = 220
    elif args.task_suite_name == "libero_cf_object":
        max_steps = 280
    elif args.task_suite_name == "libero_cf_long":
        max_steps = 500
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes, total_touches = 0, 0, 0
    # Aggregate per-condition stats across tasks when custom conditions are provided
    global_condition_stats = {}  # cond_str -> {"episodes": int, "touches": int, "successes": int}
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if task_id < args.starting_task_id:
            continue
        # Get task
        task = task_suite.get_task(task_id)
        if task_step_limit is not None:
            max_steps = task_step_limit(task_id)
        print(f"task: {task}")

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Optional: preset custom predicate checks and OR termination
        # Source of truth: per-task DB only
        selected_conditions = []
        # Resolve problem file path for lookup
        # task_bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        task_bddl_file = task.problem_folder + "/" + task.bddl_file
        logging.info(f"task_bddl_file: {task_bddl_file}")
        task_segment = os.path.basename(task_bddl_file).replace(".bddl", "")
        if conditions_db is not None:
            selected_conditions = _get_conditions_for_task(
                conditions_db,
                task_bddl_file,
            )
            logging.info(f"selected_conditions: {selected_conditions}")
        if len(selected_conditions) > 0:
            # Initialize per-condition stats for this task
            task_condition_stats = {c: {"episodes": 0, "touches": 0, "successes": 0} for c in selected_conditions}
            for c in selected_conditions:
                if c not in global_condition_stats:
                    global_condition_stats[c] = {"episodes": 0, "touches": 0, "successes": 0}
            try:
                env.set_custom_conditions(selected_conditions)
            except Exception:
                pass
            # Always enable OR-termination when custom_conditions is provided
            if conditions_db is not None:
                try:
                    env.set_success_any_conditions(selected_conditions)
                except Exception:
                    pass
            # Prepare resolution cache for subject instance -> root body name
            subject_to_body_cache = {}
            unique_subjects = []
            for cond in selected_conditions:
                subj = _condition_subject_token(cond)
                if subj and subj not in unique_subjects:
                    unique_subjects.append(subj)
            # Resolve once and print mapping for debugging
            for subj in unique_subjects:
                resolved = _resolve_instance_root_body_name(env, subj)
                subject_to_body_cache[subj] = resolved
                logging.info(f"[Mapping] subject '{subj}' -> root body '{resolved}'")

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_touches = 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask description: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            eef_points = []  # collect end-effector positions for point cloud export
            episode_trajectory = []  # collect per-step info for JSON export
            # Per-condition touch flags for this episode (when conditions are active)
            per_condition_episode_touched = {c: False for c in selected_conditions} if len(selected_conditions) > 0 else {}

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Record end-effector position for point cloud
                    try:
                        eef_pos = np.array(obs["robot0_eef_pos"], dtype=float).reshape(3)
                        eef_points.append(eef_pos)
                    except Exception:
                        pass

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]

                        if action_chunk.ndim == 1:
                            action_chunk = action_chunk.reshape(1, -1)

                        # print(f"action_chunk.shape: {action_chunk.shape}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    # Best-effort touch detection after sim step
                    try:
                        if len(selected_conditions) > 0:
                            contact_names = _get_gripper_contact_body_names(env)
                            for cond in selected_conditions:
                                if per_condition_episode_touched.get(cond, False):
                                    continue
                                subj = _condition_subject_token(cond)
                                resolved = subject_to_body_cache.get(subj) if subj else None
                                if resolved and (resolved in contact_names):
                                    per_condition_episode_touched[cond] = True
                    except Exception as e:
                        logging.error(f"Failed to record touch event: {e}")
                    

                    # Record trajectory step
                    try:
                        step_entry = {
                            "timestep": int(t),
                            "action": list(map(float, np.array(action, dtype=float).reshape(-1))),
                            "eef_pos": list(map(float, np.array(obs.get("robot0_eef_pos", []), dtype=float).reshape(-1))) if "robot0_eef_pos" in obs else [],
                            "eef_quat": list(map(float, np.array(obs.get("robot0_eef_quat", []), dtype=float).reshape(-1))) if "robot0_eef_quat" in obs else [],
                            "gripper_qpos": list(map(float, np.array(obs.get("robot0_gripper_qpos", []), dtype=float).reshape(-1))) if "robot0_gripper_qpos" in obs else [],
                        }
                        episode_trajectory.append(step_entry)
                    except Exception as e:
                        logging.error(f"Failed to record trajectory step: {e}")
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Update per-condition stats if active
            if len(selected_conditions) > 0:
                try:
                    cond_results = env.evaluate_conditions(selected_conditions)
                except Exception:
                    cond_results = {}

                success_cond = None

                # # Determine which condition(s) succeeded this episode
                # successful_conditions = [c for c in selected_conditions if bool(cond_results.get(c, False))]
                # # Choose a primary successful condition (if any) for reporting / video naming
                # success_cond = successful_conditions[0] if len(successful_conditions) > 0 else None

                # # If one condition succeeded, mark all other conditions as untouched for this episode
                # # (even if they were previously detected as touched)
                # if success_cond is not None:
                #     for c in selected_conditions:
                #         if c != success_cond and per_condition_episode_touched.get(c, False):
                #             per_condition_episode_touched[c] = False

                # Now update per-condition stats using the (possibly adjusted) touch flags
                for cond in selected_conditions:
                    # increment episode counts
                    task_condition_stats[cond]["episodes"] += 1
                    global_condition_stats[cond]["episodes"] += 1
                    # touch per condition
                    if per_condition_episode_touched.get(cond, False):
                        task_condition_stats[cond]["touches"] += 1
                        global_condition_stats[cond]["touches"] += 1
                    # success per condition (state holds at episode end)
                    if bool(cond_results.get(cond, False)):
                        task_condition_stats[cond]["successes"] += 1
                        global_condition_stats[cond]["successes"] += 1
                        success_cond = cond
                    # Debug: if success but no touch, print contact set and mapping info
                    if bool(cond_results.get(cond, False)) and not per_condition_episode_touched.get(cond, False):
                        subj = _condition_subject_token(cond)
                        resolved = subject_to_body_cache.get(subj) if subj else None
                        contact_names = _get_gripper_contact_body_names(env)
                        logging.info(f"[TouchDebug] Condition '{cond}' succeeded but no touch detected.")
                        logging.info(f"[TouchDebug] subject='{subj}' resolved_root='{resolved}' contact_roots={sorted(list(contact_names))}")


            suffix = "success" if done else "failure"
            if len(selected_conditions) > 0:
                if suffix == "success":
                    video_name = f"rollout_{task_segment}_{episode_idx:03d}_{suffix}_({success_cond}).mp4"
                else:
                    video_name = f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4"
            else:
                video_name = f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4"
            
            # Save a replay video of the episode
            imageio.mimwrite(
                pathlib.Path(video_out_path) / video_name,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Save end-effector trajectory as a point cloud (PLY)
            try:
                if len(eef_points) > 0:
                    ply_path = pathlib.Path(eef_pointcloud_out_path) / video_name.replace(".mp4", ".ply")
                    _save_points_as_ply(np.asarray(eef_points, dtype=float), ply_path)
            except Exception as e:
                # Do not fail episode if export fails
                logging.error(f"Failed to export end-effector trajectory as a point cloud: {e}")

            # Save per-episode trajectory JSON
            try:
                traj_path = pathlib.Path(trajectory_out_path) / video_name.replace(".mp4", ".json")
                if len(selected_conditions) > 0:
                    episode_summary = {
                        "task_description": str(task_description),
                        "episode_index": int(episode_idx),
                        "result": str(suffix),
                        "conditions": list(selected_conditions) if len(selected_conditions) > 0 else [],
                        "per_condition": [
                            {
                                "condition": str(cond),
                                "touched": bool(per_condition_episode_touched.get(cond, False)),
                                "success": bool(cond_results.get(cond, False)) if 'cond_results' in locals() else False,
                            }
                            for cond in (selected_conditions or [])
                        ],
                        "steps": episode_trajectory,
                    }
                else:
                    episode_summary = {
                        "task_description": str(task_description),
                        "episode_index": int(episode_idx),
                        "result": str(suffix),
                        "steps": episode_trajectory,
                    }
                episode_summary.update({
                    "task_id": int(task_id),
                    "input_modality": input_modality,
                    "original_prompt": str(task_description),
                    "sent_prompt": str(task_description),
                    "max_steps": int(max_steps),
                })
                with open(traj_path, "w") as fjson:
                    json.dump(episode_summary, fjson, indent=2)
                logging.info(f"Wrote episode trajectory to {traj_path}")
            except Exception as e:
                logging.error(f"Failed to write episode trajectory JSON: {e}")

            if len(selected_conditions) > 0:
                # Print running per-condition tallies for this task
                for cond in selected_conditions:
                    stats = task_condition_stats[cond]
                    logging.info(f"[Cond] {cond} - touches: {stats['touches']}/{stats['episodes']}  ({(stats['touches'] / stats['episodes']) * 100:.1f}%)")
                    logging.info(f"[Cond] {cond} - successes: {stats['successes']}/{stats['episodes']}  ({(stats['successes'] / stats['episodes']) * 100:.1f}%)")
            else:
                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                logging.info(f"# touches: {total_touches} ({total_touches / total_episodes * 100:.1f}%)")


        # Log final results
        # Per-condition task-level stats
        logging.info("-" * 50)
        logging.info(f"Final results for task: {task_segment}")
        if len(selected_conditions) > 0:
            for cond in selected_conditions:
                stats = task_condition_stats[cond]
                if stats["episodes"] > 0:
                    logging.info(
                        f"[Task] Condition '{cond}' - touch rate: {stats['touches']}/{stats['episodes']}  ({(stats['touches'] / stats['episodes']) * 100:.1f}%)"
                    )
                    logging.info(
                        f"[Task] Condition '{cond}' - success rate: {stats['successes']}/{stats['episodes']}  ({(stats['successes'] / stats['episodes']) * 100:.1f}%)"
                    )
        else:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current task touch rate: {float(task_touches) / float(task_episodes)}")
        logging.info("-" * 50)

    if conditions_db is None:    
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # If segmentation env is available and caller wants masking, prefer it. Fallback to OffScreenRenderEnv.
    # We infer intent from presence of camera_segmentations in args; this function is stateless otherwise.
    if SegmentationRenderEnv is not None:
        try:
            env = SegmentationRenderEnv(camera_segmentations="instance", **env_args)  # type: ignore
        except Exception:
            env = OffScreenRenderEnv(**env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _setup_logging(video_out_path: str, task_suite_name: str) -> None:
    """
    Configure logging to both console and a file alongside the rollout outputs.
    Log file path: <parent_of_video_out_path>/logs/<task_suite_name>.log
    """
    try:
        log_file = pathlib.Path(video_out_path).parent / f"{exp_name}.log"

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicate logs
        for h in list(logger.handlers):
            logger.removeHandler(h)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (append)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logging.info(f"Logging to file: {log_file}")
    except Exception as e:
        # Fallback to default basic config if anything goes wrong
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Failed to configure dual logging: {e}")


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _save_points_as_ply(points_xyz: np.ndarray, out_path: pathlib.Path) -> None:
    """Save Nx3 points as an ASCII PLY point cloud."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")
    num_points = points_xyz.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header\n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(header))
        for p in points_xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")



def _get_gripper_contact_body_names(env):
    """
    Return a set of instance (root body) names that are currently in contact with the gripper.
    We walk each contacted body's parent chain up to its root (child of world) to get the instance name,
    then return those root names. This enables exact matching against condition object instance names.
    """
    names = set()
    try:
        sim = getattr(env, "sim", None)
        if sim is None or getattr(sim, "data", None) is None or getattr(sim, "model", None) is None:
            return names
        model = sim.model
        data = sim.data

        def body_name(body_id: int) -> str:
            try:
                return model.body_id2name(int(body_id))
            except Exception:
                try:
                    return str(model.body_names[int(body_id)])
                except Exception:
                    return ""

        def is_gripper_body(name: str) -> bool:
            n = name.lower()
            return ("gripper" in n) or ("finger" in n) or ("hand" in n) or ("panda_hand" in n)

        def root_body_id(bid: int) -> int:
            """
            Ascend the body tree until reaching a direct child of world (parent id == 0).
            Return that root body id. Defensive against invalid indices.
            """
            try:
                parent = int(model.body_parentid[bid])
            except Exception:
                return bid
            # body 0 is world
            while parent > 0:
                bid = parent
                try:
                    parent = int(model.body_parentid[bid])
                except Exception:
                    break
            return bid

        ncon = int(getattr(data, "ncon", 0))
        for i in range(ncon):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
                b1_name = body_name(b1)
                b2_name = body_name(b2)
                if is_gripper_body(b1_name) and b2_name:
                    rb2 = root_body_id(b2)
                    names.add(body_name(rb2))
                elif is_gripper_body(b2_name) and b1_name:
                    rb1 = root_body_id(b1)
                    names.add(body_name(rb1))
            except Exception:
                continue
        return names
    except Exception:
        return names

def _resolve_instance_root_body_name(env, instance_name: str):
    """
    Best-effort resolution from a BDDL instance name to the corresponding root body name in MuJoCo.
    Strategy:
      1) If a body with exact name exists -> return it.
      2) Otherwise, find any body whose name starts with the instance name, and return the most common
         root body among those matches. If none, return None.
    """
    try:
        sim = getattr(env, "sim", None)
        if sim is None or getattr(sim, "model", None) is None:
            return None
        model = sim.model
        # 1) exact match
        try:
            _ = model.body_name2id(instance_name)
            return instance_name
        except Exception:
            pass

        # Helpers
        def body_name(body_id: int) -> str:
            try:
                return model.body_id2name(int(body_id))
            except Exception:
                try:
                    return str(model.body_names[int(body_id)])
                except Exception:
                    return ""

        def root_body_id(bid: int) -> int:
            try:
                parent = int(model.body_parentid[bid])
            except Exception:
                return bid
            while parent > 0:
                bid = parent
                try:
                    parent = int(model.body_parentid[bid])
                except Exception:
                    break
            return bid

        # 2) prefix search
        matches = []
        try:
            nbody = int(model.nbody)
        except Exception:
            nbody = 0
        for bid in range(nbody):
            name = body_name(bid)
            if name.startswith(instance_name):
                matches.append(bid)
        if not matches:
            return None
        # Count roots
        from collections import Counter
        roots = [root_body_id(b) for b in matches]
        root_counts = Counter(roots)
        best_root, _ = root_counts.most_common(1)[0]
        return body_name(best_root)
    except Exception:
        return None


def _condition_subject_token(cond_str: str):
    """
    Parse a custom condition string and return the subject object token (first argument after predicate).
    Example:
      "on cookies_1 plate_1" -> "cookies_1"
      "open drawer_1" -> "drawer_1"
    """
    try:
        tokens = str(cond_str).strip().split()
        if len(tokens) >= 2:
            return tokens[1]
        return None
    except Exception:
        return None


def _load_custom_conditions_db(task_suite_name: str):
    """
    Load per-task custom conditions from a JSON file.
    Accepted schema (single format):
      {
        ".../bddl_a": ["on cookies_1 plate_1", "on akita_black_bowl_1 plate_1"],
        ".../bddl_b": ["in cream_cheese_1 bowl_1"]
      }
    Key must match the relative problem file string used at runtime, e.g.:
      "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
    Returns a dict: problem_file_relative_str -> List[str]
    """
    path = pathlib.Path(get_libero_path("conditions")) / f"{task_suite_name}.json"
    
    if not path.exists():
        logging.error(f"custom_conditions does not exist: {path}")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load custom conditions JSON from {path}: {e}")
        return None

    result: dict[str, list[str]] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            key = str(k)
            # v must be a list of strings (conditions)
            if isinstance(v, list):
                # Normalize entries to strings
                conds = [str(x) for x in v]
                result[key] = conds
            else:
                logging.warning(f"Ignoring entry for {k}: expected list of strings, got {type(v)}")
    else:
        logging.warning(f"Unsupported custom conditions JSON format at {path} (expected dict mapping to list[str])")
        return None
    logging.info(f"Loaded per-task custom conditions (single-schema) for {len(result)} tasks from {path}")
    return result


def _get_conditions_for_task(db, problem_key: str):
    """
    Resolve conditions for a given task. Prefer problem_file match; if not found,
    return an empty list.
    """
    if not db:
        raise ValueError("No custom conditions database loaded.")
    key = str(problem_key)
    if key not in db:
        raise ValueError(f"No custom conditions found for task problem_file: {key}")
    conds_value = db[key]
    if not isinstance(conds_value, list) or len(conds_value) == 0:
        raise ValueError(f"Custom conditions for task are missing or empty: {key}")
    # Defensive copy
    return list(conds_value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
