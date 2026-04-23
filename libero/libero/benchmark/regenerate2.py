import os
import numpy as np
import torch
from libero.libero import get_libero_path, benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

SUITE = "libero_cf_spatial_focused"
# TASK = "03-pick_up_the_cookie_box_and_place_it_on_the_plate"
BOWL = "akita_black_bowl_1_main"  # instance name

tasks = [
    "04-pick_up_the_cookie_box_and_place_it_on_the_plate",
    "05-pick_up_the_ramekin_and_place_it_on_the_plate",
    "06-pick_up_the_ramekin_and_place_it_on_the_plate",
    "07-pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "08-pick_up_the_cookie_box_and_place_it_on_the_plate",
    "09-pick_up_the_cookie_box_and_place_it_on_the_plate",
    "10-pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "11-pick_up_the_ramekin_and_place_it_on_the_plate",
    "12-pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "13-pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    "14-pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    "15-pick_up_the_ramekin_and_place_it_on_the_plate",
]

for TASK in tasks:

    bddl = os.path.join(get_libero_path("bddl_files"), SUITE, f"{TASK}.bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=True)

    m = env.sim.model
    nq, nv = m.nq, m.nv
    qpos_off = 0
    qvel_off = nq  # qvel starts after qpos in the flattened state

    # body_names = [m.body_id2name(i) for i in range(m.nbody)]
    # geom_names = [m.geom_id2name(i) for i in range(m.ngeom)]
    # print("Bodies:", body_names)
    # print("length of bodies:", len(body_names))
    # print("Geoms:", geom_names)
    # print("length of geoms:", len(geom_names))

    def _joint_sizes(jtype: int):
        # 0: free (7 qpos, 6 qvel), 1: ball (4,3), 2: slide (1,1), 3: hinge (1,1)
        return (7, 6) if jtype == 0 else (4, 3) if jtype == 1 else (1, 1)

    def slices_for_instance(instance_name: str):
        """Return list of ((qpos_lo, qpos_hi), (qvel_lo, qvel_hi)) for all joints on this body."""
        try:
            body_id = m.body_name2id(instance_name)
        except Exception:
            return []

        j = int(m.body_jntadr[body_id])
        if j < 0:
            return []

        out = []
        # Collect joints belonging to this body
        while j < m.njnt and int(m.jnt_bodyid[j]) == body_id:
            pos_size, vel_size = _joint_sizes(int(m.jnt_type[j]))
            qp0 = int(m.jnt_qposadr[j]); qp1 = qp0 + pos_size
            qv0 = int(m.jnt_dofadr[j]);  qv1 = qv0 + vel_size
            out.append(((qp0, qp1), (qv0, qv1)))
            j += 1
        return out

    bowl_slices = slices_for_instance(BOWL)
    assert bowl_slices, f"No joints found for {BOWL}. Check instance name."

    def modify_bowl_only(flat_state: np.ndarray, mode="remove", teleport_xyz=np.array([0.0, 0.0, 0.5])):
        """Return a copy of flat_state with ONLY bowl qpos/qvel changed."""
        state = flat_state.copy()
        for (qp0, qp1), (qv0, qv1) in bowl_slices:
            # Free joints are typical for movable objects; if not free, this still works by slice
            qpos = state[qpos_off + qp0 : qpos_off + qp1]
            qvel = state[qvel_off + qv0 : qvel_off + qv1]

            if qp1 - qp0 == 7:  # free joint
                if mode == "remove":
                    qpos[:3] = np.array([0.0, 0.0, -10.0])  # move far below
                elif mode == "teleport":
                    qpos[:3] = teleport_xyz  # keep quaternion as-is
                qvel[:] = 0.0
            else:
                # Hinge/slide/ball: zero velocities; optionally set pos to a neutral value
                if mode == "remove":
                    qvel[:] = 0.0
                elif mode == "teleport":
                    qvel[:] = 0.0  # nothing else to do without a free joint

            state[qpos_off + qp0 : qpos_off + qp1] = qpos
            state[qvel_off + qv0 : qvel_off + qv1] = qvel
        return state

    # Example: load existing init states, modify only the bowl, save back
    init_dir = os.path.join(get_libero_path("init_states"), SUITE)
    in_path  = os.path.join(init_dir, f"{TASK}.pruned_init")
    out_path = os.path.join(init_dir, f"{TASK}.pruned_init")  # overwrite; or write to a new file if you prefer

    orig = torch.load(in_path)
    orig_np = orig.detach().cpu().numpy() if isinstance(orig, torch.Tensor) else np.asarray(orig)

    new_np = np.stack([modify_bowl_only(s, mode="remove") for s in orig_np]).astype(np.float32)
    torch.save(torch.from_numpy(new_np), out_path)
    print("Saved:", new_np.shape, "to", out_path)

    # Optional sanity: verify env can load one modified state
    env.regenerate_obs_from_state(new_np[0])