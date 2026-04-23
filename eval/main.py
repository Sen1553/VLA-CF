import collections
import dataclasses
import logging
import math
import pathlib
from datetime import datetime
import traceback
import json

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import torch
from PIL import Image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    global exp_name
    exp_name = f"{args.task_suite_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    save_path = f"rollouts/{exp_name}"
    video_out_path = f"{save_path}/videos"
    eef_pointcloud_out_path = f"{save_path}/eef_pointclouds"
    trajectory_out_path = f"{save_path}/trajectories"
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(eef_pointcloud_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trajectory_out_path).mkdir(parents=True, exist_ok=True)

    # Setup logging to both console and file
    _setup_logging(video_out_path, args.task_suite_name)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes, total_touches = 0, 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Identify a single goal condition from the BDDL (first goal predicate)
        primary_condition = _get_primary_goal_condition(env)
        subject_to_body_cache = {}
        if primary_condition:
            logging.info(f"Tracking goal condition (Bddl): {primary_condition}")
            subj = _condition_subject_token(primary_condition)
            if subj:
                resolved = _resolve_instance_root_body_name(env, subj)
                subject_to_body_cache[subj] = resolved
                logging.info(f"[Mapping] subject '{subj}' -> root body '{resolved}'")
        else:
            logging.info("No goal condition found in BDDL; touch-rate tracking disabled for this task.")

        # Start episodes
        task_episodes, task_successes, task_touches = 0, 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            # replay_flow_images = []
            episode_touched = False
            eef_points = []
            episode_trajectory = []

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

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            # "observation/image": np.zeros_like(img),
                            # "observation/wrist_image": np.zeros_like(wrist_img),
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
                        outputs = client.infer(element)
                        action_chunk = outputs["actions"]

                        if action_chunk.ndim == 1:
                            action_chunk = action_chunk.reshape(1, -1)
                        
                        # print(f"action_chunk.shape: {action_chunk.shape}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    # replay_flow_images.append(decoded_flow_image)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    # Best-effort touch detection after sim step
                    if primary_condition:
                        try:
                            contact_names = _get_gripper_contact_body_names(env)
                            subj = _condition_subject_token(primary_condition)
                            resolved = subject_to_body_cache.get(subj) if subj else None
                            if resolved and (resolved in contact_names):
                                episode_touched = True
                        except Exception as e:
                            logging.error(f"Failed to record touch event: {e}")
                    # Record end-effector position for point cloud export
                    try:
                        eef_pos = np.array(obs.get("robot0_eef_pos", []), dtype=float).reshape(3)
                        eef_points.append(eef_pos)
                    except Exception:
                        pass
                    # Record per-step trajectory
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
                    logging.error("Traceback:\n%s", traceback.format_exc())
                    break

            task_episodes += 1
            total_episodes += 1

            if primary_condition:
                task_touches += int(episode_touched)
                total_touches += int(episode_touched)

                try:
                    cond_results = env.evaluate_conditions([primary_condition])
                except Exception:
                    cond_results = {}
                cond_success = bool(cond_results.get(primary_condition, False))

                if cond_success and not episode_touched:
                    subj = _condition_subject_token(primary_condition)
                    resolved = subject_to_body_cache.get(subj) if subj else None
                    contact_names = _get_gripper_contact_body_names(env)
                    logging.info(f"[TouchDebug] Condition '{primary_condition}' succeeded but no touch detected.")
                    logging.info(f"[TouchDebug] subject='{subj}' resolved_root='{resolved}' contact_roots={sorted(list(contact_names))}")

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            # concat replay_images and replay_flow_images on the width dimension
            # video_frames = np.concatenate([replay_images, replay_flow_images], axis=2)
            video_frames = replay_images
            # print(f"video_frames shape: {video_frames.shape}")
            imageio.mimwrite(
                pathlib.Path(video_out_path) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4",
                [np.asarray(x) for x in video_frames],
                fps=10,
            )

            # Save end-effector trajectory as a point cloud (PLY)
            try:
                if len(eef_points) > 0:
                    ply_path = pathlib.Path(eef_pointcloud_out_path) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.ply"
                    _save_points_as_ply(np.asarray(eef_points, dtype=float), ply_path)
            except Exception as e:
                logging.error(f"Failed to export end-effector trajectory as a point cloud: {e}")

            # Save per-episode trajectory JSON
            try:
                traj_path = pathlib.Path(trajectory_out_path) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.json"
                episode_summary = {
                    "task_description": str(task_description),
                    "episode_index": int(episode_idx),
                    "result": str(suffix),
                    "condition": primary_condition if primary_condition else None,
                    "touched": bool(episode_touched) if primary_condition else False,
                    "steps": episode_trajectory,
                }
                with open(traj_path, "w") as fjson:
                    json.dump(episode_summary, fjson, indent=2)
                logging.info(f"Wrote episode trajectory to {traj_path}")
            except Exception as e:
                logging.error(f"Failed to write episode trajectory JSON: {e}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if primary_condition:
                logging.info(f"# touches: {total_touches} ({total_touches / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        if primary_condition:
            logging.info(f"Current task touch rate: {float(task_touches) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if total_episodes > 0:
        logging.info(f"Total touch rate: {float(total_touches) / float(total_episodes)}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _setup_logging(video_out_path: str, task_suite_name: str) -> None:
    """
    Configure logging to both console and a file alongside the rollout outputs.
    Log file path: <parent_of_video_out_path>/<exp_name>.log
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


def _get_primary_goal_condition(env):
    """
    Extract the first goal condition from the BDDL file and normalize it into a
    single string, e.g., "on cookies_1 plate_1".
    """
    try:
        parsed = getattr(env, "env", None)
        goal_state = getattr(parsed, "parsed_problem", {}).get("goal_state", [])
        if isinstance(goal_state, list) and len(goal_state) > 0:
            cond = goal_state[0]
            if isinstance(cond, str):
                return cond.strip().lower()
            if isinstance(cond, (list, tuple)):
                tokens = [str(cond[0]).lower()] + [str(x) for x in cond[1:]]
                return " ".join(tokens)
    except Exception as e:
        logging.error(f"Failed to extract goal condition from BDDL: {e}")
    return None


def _condition_subject_token(cond_str: str):
    """
    Parse a condition string and return the subject object token (first argument
    after predicate).
    Example: "on cookies_1 plate_1" -> "cookies_1"
    """
    try:
        tokens = str(cond_str).strip().split()
        if len(tokens) >= 2:
            return tokens[1]
        return None
    except Exception:
        return None


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
