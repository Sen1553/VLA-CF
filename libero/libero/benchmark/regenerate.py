import os
import numpy as np
import torch
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
try:
    # Optional import for segmentation-enabled env
    from libero.libero.envs import SegmentationRenderEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    SegmentationRenderEnv = None

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
SEED = 7


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

def _modify_target_instance(env_wrapper):
    if OBJECT_UPDATE_MODE == "remove":
        _move_instance_out_of_scene(env_wrapper, TARGET_INSTANCE_NAME)
    elif OBJECT_UPDATE_MODE == "teleport":
        _teleport_instance(env_wrapper, TARGET_INSTANCE_NAME, TELEPORT_XYZ)


benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_cf"]()

task_id = 0
task = task_suite.get_task(task_id)
initial_states = task_suite.get_task_init_states(task_id)
print(initial_states.shape)

env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)

env.reset()
obs = env.set_init_state(initial_states[0])
