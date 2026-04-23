import os
import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Suite and task
SUITE = "libero_cf_long"
TASK_BASENAME = "03-put_the_alphabet_soup_in_the_basket"

# Object instance name to modify (as used by LIBERO instance naming, e.g., "akita_black_bowl_1")
TARGET_INSTANCE_NAME = "alphabet_soup_1"

# If True and an existing init file is present, reuse those states and ONLY modify the
# target object's state while keeping all other objects unchanged. If this fails due to
# scene mismatch, the script will fall back to regenerating the same number of states.
REUSE_AND_MODIFY_EXISTING = True

# When not reusing, we will try to infer the number of states from an existing file if
# available. If not found, fall back to this default.
DEFAULT_NUM_STATES = 100

# How to modify the target object:
# - "remove": move it far below the workspace (out of sight / no contact)
# - "teleport": place at a specific xyz (keep current orientation)
OBJECT_UPDATE_MODE = "remove"  # or "teleport"
TELEPORT_XYZ = np.array([0.0, 0.0, 0.5])  # used when OBJECT_UPDATE_MODE == "teleport"


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
bddl_path = os.path.join(get_libero_path("bddl_files"), SUITE, f"{TASK_BASENAME}.bddl")
out_dir = os.path.join(get_libero_path("init_states"), SUITE)
os.makedirs(out_dir, exist_ok=True)
init_state_path = os.path.join(out_dir, f"{TASK_BASENAME}-old.pruned_init")
out_path = os.path.join(out_dir, f"{TASK_BASENAME}.pruned_init")


def _move_instance_out_of_scene(env_wrapper, instance_name: str, z_out: float = -10.0) -> bool:
    """Move a free-jointed instance far below the workspace. Returns True if succeeded.

    This attempts to find the root joint from the instance's body and set its qpos/qvel.
    Works for free-jointed movable objects (typical for manipulable items).
    """
    sim = env_wrapper.env.sim
    model = sim.model
    data = sim.data

    try:
        body_id = model.body_name2id(instance_name)
    except Exception:
        return False

    jntadr = int(model.body_jntadr[body_id])
    if jntadr < 0:
        return False

    qpos_adr = int(model.jnt_qposadr[jntadr])
    qvel_adr = int(model.jnt_dofadr[jntadr])
    jnt_type = int(model.jnt_type[jntadr])

    # 0 == mjJNT_FREE in MuJoCo
    if jnt_type != 0:
        return False

    # Copy out, modify, and set back
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    # Set translation far below, keep orientation unchanged
    qpos[qpos_adr : qpos_adr + 3] = np.array([0.0, 0.0, z_out], dtype=float)
    qvel[qvel_adr : qvel_adr + 6] = 0.0

    sim.set_state_from_flattened(np.concatenate([qpos, qvel]))
    sim.forward()
    return True


def _teleport_instance(env_wrapper, instance_name: str, xyz: np.ndarray) -> bool:
    """Teleport instance with a free joint to the given xyz (keep current orientation)."""
    sim = env_wrapper.env.sim
    model = sim.model
    data = sim.data

    try:
        body_id = model.body_name2id(instance_name)
    except Exception:
        return False

    jntadr = int(model.body_jntadr[body_id])
    if jntadr < 0:
        return False

    qpos_adr = int(model.jnt_qposadr[jntadr])
    qvel_adr = int(model.jnt_dofadr[jntadr])
    jnt_type = int(model.jnt_type[jntadr])

    if jnt_type != 0:
        return False

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    qpos[qpos_adr : qpos_adr + 3] = np.asarray(xyz, dtype=float)
    qvel[qvel_adr : qvel_adr + 6] = 0.0

    sim.set_state_from_flattened(np.concatenate([qpos, qvel]))
    sim.forward()
    return True


def _modify_target_instance(env_wrapper):
    if OBJECT_UPDATE_MODE == "remove":
        _move_instance_out_of_scene(env_wrapper, TARGET_INSTANCE_NAME)
    elif OBJECT_UPDATE_MODE == "teleport":
        _teleport_instance(env_wrapper, TARGET_INSTANCE_NAME, TELEPORT_XYZ)


def main():
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True
    )

    # Try to reuse existing init states
    if REUSE_AND_MODIFY_EXISTING and os.path.exists(out_path):
        original = torch.load(init_state_path)
        # original = original.detach().cpu().numpy()
        print("original.shape:", original.shape)
        new_states = []
        reuse_ok = True
        for i in range(original.shape[0]):
            # try:
            # Set to old state, then ONLY modify the target instance
            env.regenerate_obs_from_state(original[i])
            _modify_target_instance(env)
            new_states.append(env.get_sim_state())
            # except Exception:
            #     reuse_ok = False
            #     break

        if reuse_ok and len(new_states) == original.shape[0]:
            init_states = torch.from_numpy(np.stack(new_states)).float()
            torch.save(init_states, out_path)
            print(f"Rewrote (reused) {init_states.shape} to {out_path}")
            return
        else:
            print(
                "[warn] Reusing existing states failed (likely scene changed). Falling back to regeneration."
            )
            inferred_num_states = int(original.shape[0])
    else:
        # Infer number of states from existing file if present
        if os.path.exists(out_path):
            try:
                existing = torch.load(out_path)
                inferred_num_states = int(existing.shape[0])
            except Exception:
                inferred_num_states = DEFAULT_NUM_STATES
        else:
            inferred_num_states = DEFAULT_NUM_STATES

    # Regenerate a fresh set with the inferred count
    states = []
    for i in range(inferred_num_states):
        env.reset()
        env.set_init_state(env.get_init_state() )
        _modify_target_instance(env)
        states.append(env.get_sim_state())

    init_states = torch.from_numpy(np.stack(states)).float()
    torch.save(init_states, out_path)
    print(f"Saved {init_states.shape} to {out_path}")


if __name__ == "__main__":
    main()