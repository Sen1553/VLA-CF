# LIBERO-CF: Counterfactual Robustness Evaluation of Vision-Language-Action Models

LIBERO-CF is an evaluation benchmark built on top of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for testing whether Vision-Language-Action (VLA) models can correctly follow counterfactual instructions.

By evaluating a set of plausible conditions for each task, LIBERO-CF probes whether a policy truly grounds language in the scene rather than relying on memorized behaviors.

We measure two metrics per condition:
- **Touch rate**: whether the end-effector contacts the object associated with the condition
- **Success rate**: whether the condition is successfully completed

---

## Task Suites

| Suite | Tasks | Description |
|---|---|---|
| `libero_cf_spatial` | 15 | Forked from LIBERO-Spatial, up to 4 conditions per task |
| `libero_cf_spatial_focused` | 15 | Focused scene variant of `libero_cf_spatial` |
| `libero_cf_object` | 10 | Forked from LIBERO-Object, up to 4 conditions per task |
| `libero_cf_long` | 10 | Forked from LIBERO-10, long-horizon manipulation |
| `libero_cf_ood` | 15 | Out-of-distribution objects (coke can, beer, cup, apple, banana) not seen in LIBERO training |

---

## Dataset

LIBERO-CF does not require **additional data collection** beyond the standard [LIBERO dataset](https://huggingface.co/datasets/physical-intelligence/libero) (LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10). All task suites are designed to evaluate models that have been fine-tuned on the standard LIBERO dataset. The benchmark probes whether those fine-tuned policies generalize beyond their training scenarios.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yuffish/libero_cf.git
cd libero_cf
```

Create and activate a virtual environment with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.8 --prompt libero_cf .venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip sync requirements.txt requirements_libero.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e .
```

---

## Evaluation

### π0 / π0.5 (openpi)

In the **openpi** environment, serve the policy:

```bash
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py --env LIBERO
```

In a separate terminal, run the evaluation:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export LIBERO_CONFIG_PATH=$PWD/.libero

# Replace <suite> with one of: libero_cf_spatial, libero_cf_object, libero_cf_long,
#   libero_cf_spatial_focused, libero_cf_ood
python eval/main_cf.py \
    --args.task_suite_name <suite> \
    --args.port <port>
```

### OpenVLA-OFT

In the **openvla-oft** environment, copy `packages/` into the root of your OpenVLA-OFT repository, then install:

```bash
conda activate openvla-oft
pip install -e packages/openpi-client
pip install tyro
```

Serve the policy:

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft.py \
    --model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name <suite> \
    --port <port>
```

In a separate terminal, run the evaluation (use `--args.replan_steps 8` for OpenVLA-OFT):

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export LIBERO_CONFIG_PATH=$PWD/.libero

python eval/main_cf.py \
    --args.task_suite_name <suite> \
    --args.port <port> \
    --args.replan_steps 8
```

Rollouts are saved to `rollouts/<suite>-<timestamp>/`.

### Visualization

```bash
python eval/batch_visualize.py --rollouts_dir rollouts/<your_rollouts_dir>
```

Outputs per-task videos, end-effector trajectories, and gripper-close heatmaps.

---


## Example Output

All examples below use **50 rollouts per task** on an NVIDIA RTX 6000 (~17 min per task).

### π0.5 on `libero_cf_spatial`

Task: `01-pick_up_the_cookie_box_and_place_it_on_the_plate`

```
Condition 'on cookies_1 plate_1'          touch rate: 35/50 (70.0%)   success rate: 25/50 (50.0%)
Condition 'on akita_black_bowl_1 plate_1' touch rate: 16/50 (32.0%)   success rate: 15/50 (30.0%)
```

### π0.5 on `libero_cf_spatial_focused`

Task: `01-pick_up_the_cookie_box_and_place_it_on_the_plate`

```
Condition 'on cookies_1 plate_1'           touch rate: 49/50 (98.0%)   success rate: 33/50 (66.0%)
Condition 'on akita_black_bowl_2 plate_1'  touch rate:  1/50  (2.0%)   success rate:  0/50  (0.0%)
```

### π0.5 on `libero_cf_long`

Task: `01-put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet`

```
Condition 'on wine_bottle_1 white_cabinet_1_bottom_region'        touch rate: 34/50 (68.0%)   success rate: 2/50 (4.0%)
Condition 'on akita_black_bowl_1 white_cabinet_1_bottom_region'   touch rate: 21/50 (42.0%)   success rate: 0/50 (0.0%)
```

---

## Results

Success rates (%) on LIBERO-CF. **F.** (Faithful) = success rate on the instructed object; **B.** (Biased) = success rate on training-task objects. Higher F. and lower B. is better.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Metric</th>
      <th colspan="2">CF-Spatial</th>
      <th colspan="2">CF-Object</th>
      <th colspan="2">CF-Long</th>
      <th colspan="2">CF-OOD</th>
      <th colspan="2">Average</th>
    </tr>
    <tr>
      <th>F.↑</th><th>B.↓</th>
      <th>F.↑</th><th>B.↓</th>
      <th>F.↑</th><th>B.↓</th>
      <th>F.↑</th><th>B.↓</th>
      <th>F.↑</th><th>B.↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">OpenVLA-OFT</td>
      <td>Grounding</td>
      <td>6.8</td><td>84.7</td>
      <td>2.0</td><td>99.6</td>
      <td>4.0</td><td>62.9</td>
      <td>6.0</td><td>87.3</td>
      <td>4.7</td><td>83.6</td>
    </tr>
    <tr>
      <td>Success</td>
      <td>1.1</td><td>80.7</td>
      <td>0.0</td><td>97.6</td>
      <td>0.2</td><td>54.4</td>
      <td>0.1</td><td>81.7</td>
      <td>0.4</td><td>78.6</td>
    </tr>
    <tr>
      <td rowspan="2">π0</td>
      <td>Grounding</td>
      <td>70.9</td><td>37.1</td>
      <td>6.0</td><td>59.4</td>
      <td>20.4</td><td>83.1</td>
      <td>18.0</td><td>72.4</td>
      <td>28.8</td><td>63.0</td>
    </tr>
    <tr>
      <td>Success</td>
      <td>34.4</td><td>19.1</td>
      <td>0.0</td><td>45.0</td>
      <td>1.1</td><td>63.1</td>
      <td>2.7</td><td>52.9</td>
      <td>9.6</td><td>45.0</td>
    </tr>
    <tr>
      <td rowspan="2">π0.5</td>
      <td>Grounding</td>
      <td>39.3</td><td>61.3</td>
      <td>11.4</td><td>68.8</td>
      <td>51.6</td><td>58.4</td>
      <td>20.7</td><td>74.0</td>
      <td>30.8</td><td>65.6</td>
    </tr>
    <tr>
      <td>Success</td>
      <td>24.4</td><td>56.9</td>
      <td>5.8</td><td>66.8</td>
      <td>15.8</td><td>50.4</td>
      <td>6.9</td><td>69.6</td>
      <td>13.2</td><td>60.9</td>
    </tr>
  </tbody>
</table>

---

## Key Structure

```
libero_cf/
├── eval/
│   ├── main_cf.py            # Main evaluation script
│   ├── batch_visualize.py    # Batch visualization of saved rollouts
│   └── visualize.py          # Per-rollout trajectory and heatmap rendering
├── libero/
│   └── libero/
│       ├── bddl_files/       # Task definitions (.bddl) for each suite
│       ├── conditions/       # Per-task counterfactual condition specs (.json)
│       └── init_files/       # Initial simulator states per suite
├── packages/
│   └── openpi-client/        # WebSocket client for openpi-compatible servers
├── scripts/                  # Dataset and task creation utilities
```

---

## Counterfactual Action Guidance

Counterfactual action guidance provides an inference-time scheme to mitigate counterfactual failure, using a jointly served conditioned and unconditioned policy. We provide example implementations for openpi (**/openpi-cf**) and OpenVLA-OFT (**/openvla-oft-cf**) to serve CAG policies. Evaluation can be run using the same script described above.

### π0 / π0.5 (openpi)

Serve the CAG policy in the **openpi** environment.

#### TF version (training-free)
```bash
# pi0
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_TF.py \
    --port <port> \
    --guidance_scale 2.0 \
    policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi0_libero


# pi05
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_TF.py \
    --port <port> \
    --guidance_scale 2.0 \
    policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

#### VA version (with unconditioned model)

Download the unconditioned pi0 from [yuffish/pi0_libero_unconditioned](https://huggingface.co/yuffish/pi0_libero_unconditioned) and pi05 from [yuffish/pi05_libero_unconditioned](https://huggingface.co/yuffish/pi05_libero_unconditioned).

```bash
# pi0
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_VA.py \
    --guidance_scale 1.5 \
    --port <port> \
    policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi0_libero \
    uncond-policy:checkpoint \
        --uncond_policy.config=pi0_libero_unconditioned \
        --uncond_policy.dir=/path/to/pi0_libero_unconditioned


# pi05
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_VA.py \
    --guidance_scale 1.5 \
    --port <port> \
    policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero \
    uncond-policy:checkpoint \
        --uncond_policy.config=pi05_libero_unconditioned \
        --uncond_policy.dir=/path/to/pi05_libero_unconditioned
```

### OpenVLA-OFT

Serve the CAG policy in the **openvla-oft** environment.

#### TF version (training-free)

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft_TF.py \
    --model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name <suite> \
    --guidance_scale 2.0 \
    --port <port>
```

#### VA version (with unconditioned model)

Download the unconditioned OpenVLA-OFT from [yuffish/openvla_oft_libero_unconditioned](https://huggingface.co/yuffish/openvla_oft_libero_unconditioned).

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft_VA.py \
    --conditioned_model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --unconditioned_model_path /path/to/openvla-oft-unconditioned \
    --task_suite_name <suite> \
    --guidance_scale 2.0 \
    --port <port>
```

---

## Citation

If you find our work useful, please cite:

```bibtex
@article{liu2023libero,
  title={Libero: Benchmarking knowledge transfer for lifelong robot learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={44776--44791},
  year={2023}
}

@article{fang2026vision,
  title={When vision overrides language: Evaluating and mitigating counterfactual failures in vlas},
  author={Fang, Yu and Feng, Yuchun and Jing, Dong and Liu, Jiaqi and Yang, Yue and Wei, Zhenyu and Szafir, Daniel and Ding, Mingyu},
  journal={arXiv preprint arXiv:2602.17659},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE). Task assets follow the licensing terms of the original LIBERO dataset.
