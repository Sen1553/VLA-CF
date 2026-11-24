<h1 align="center">When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs</h1>

<p align="center"><strong>CoRL 2026 (Spotlight)</strong></p>

<p align="center">
  <a href="https://vla-cf.github.io/"><img src="https://img.shields.io/badge/Project-Website-2ea44f.svg" alt="Project Website"></a>
  <a href="https://arxiv.org/abs/2602.17659"><img src="https://img.shields.io/badge/arXiv-2602.17659-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/VLA-CF/models"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffd21e.svg" alt="Models"></a>
</p>

<p align="center">
  <a href="https://yuffish.github.io/">Yu Fang</a>,
  <a href="https://cathyf9600.github.io/">Yuchun Feng</a>,
  <a href="https://timsty1.github.io/">Dong Jing</a>,
  <a href="https://jiaaqiliu.github.io/">Jiaqi Liu</a>,
  <a href="https://yy-gx.github.io/">Yue Yang</a>,
  <a href="https://zhenyuwei2003.github.io/">Zhenyu Wei</a>,
  <a href="https://danszafir.com/">Daniel Szafir</a>,
  <a href="https://dingmyu.github.io/">Mingyu Ding</a>
  <br/>
  University of North Carolina at Chapel Hill
</p>

This repository contains the two parts of our paper:

- **LIBERO-CF**: a benchmark that tests whether Vision-Language-Action (VLA) models follow counterfactual instructions.
- **Counterfactual Action Guidance (CAG)**: an inference-time method that mitigates counterfactual failures in VLAs.

## Contents

- [Part 1: LIBERO-CF](#part-1-libero-cf)
- [Part 2: Counterfactual Action Guidance (CAG)](#part-2-counterfactual-action-guidance-cag)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Part 1: LIBERO-CF

LIBERO-CF is a counterfactual benchmark built on top of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for testing whether Vision-Language-Action (VLA) models can correctly follow counterfactual instructions. By evaluating a set of plausible conditions for each task, LIBERO-CF probes whether a policy truly grounds language in the scene rather than relying on memorized behaviors.

The conditions for each task are defined in [`libero/libero/conditions/`](libero/libero/conditions/). The evaluation logs two metrics for every condition:
- **Touch rate**: whether the gripper touches the object associated with the condition. We use this contact to indicate whether the policy follows the instruction for that condition
- **Success rate**: whether the condition is successfully completed

The paper reports results for two of these conditions per task: the instructed (counterfactual) task and the training task. The **grounding rate** is the touch rate of the corresponding condition.

### Task Suites

| Suite | Tasks | Description |
|---|---|---|
| `libero_cf_spatial` | 15 | Forked from LIBERO-Spatial, up to 4 conditions per task |
| `libero_cf_spatial_focused` | 15 | Focused scene variant of `libero_cf_spatial`: training-task object removed in each scene |
| `libero_cf_object` | 10 | Forked from LIBERO-Object; local evaluation now checks all 6 movable objects going into the basket |
| `libero_cf_long` | 10 | Forked from LIBERO-10, long-horizon manipulation |
| `libero_cf_ood` | 15 | Based on LIBERO-Spatial, with out-of-distribution objects (coke can, beer, cup, apple, banana) not seen in LIBERO training |
| `libero_cf_probe` (`libero-cf-probe`) | 7 | Independent, fixed-order selection of 3 CF-Object and 4 CF-Spatial tasks |

The **libero-cf-probe** suite has its own BDDL files, initial states, and condition
JSON under the canonical name `libero_cf_probe`. Both spellings select the same
suite. Its copied task names, conditions, and initial states are preserved. The
Object condition expansion described below does not change these independent
Probe copies. Evaluation follows this order (task IDs are zero-based):

| Task ID | Source suite | Task | Max steps |
|---|---|---|---|
| 0 | `libero_cf_object` | `05-pick_up_the_milk_and_place_it_in_the_basket` | 280 |
| 1 | `libero_cf_object` | `02-pick_up_the_butter_and_place_it_in_the_basket` | 280 |
| 2 | `libero_cf_object` | `04-pick_up_the_salad_dressing_and_place_it_in_the_basket` | 280 |
| 3 | `libero_cf_spatial` | `05-pick_up_the_ramekin_and_place_it_on_the_plate` | 220 |
| 4 | `libero_cf_spatial` | `13-pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 220 |
| 5 | `libero_cf_spatial` | `01-pick_up_the_cookie_box_and_place_it_on_the_plate` | 220 |
| 6 | `libero_cf_spatial` | `02-pick_up_the_ramekin_and_place_it_on_the_plate` | 220 |

After starting a compatible policy server and activating the benchmark environment
as described below, evaluate the probe suite from the repository root:

```bash
python eval/main_cf.py --args.task_suite_name libero-cf-probe --args.port 8000
```

Rollouts are saved under `rollouts/libero_cf_probe-<timestamp>/`, or
`rollouts/libero_cf_probe-<modality>-<timestamp>/` when the server reports a
verified input modality. The new suite
does not define new model normalization statistics; model-server normalization
must still match its checkpoint and the source tasks (Object or Spatial).

### Following / deviation metrics (local extension)

With custom conditions enabled (the default), `libero_cf_object`,
`libero_cf_spatial`, and `libero_cf_ood` use the
`same_destination_first_success_v1` protocol. Object monitors all six non-basket
objects going into the basket; Spatial and OOD each monitor four non-plate
objects going onto the plate. This is a finite **same-destination** catalog,
not an enumeration of every feasible manipulation in the scene. Long, Focused,
and Probe retain their existing evaluation paths.

The original `In`/`On` predicates and first-candidate-success OR termination are
retained. Limits remain 280 policy actions for Object and 220 for Spatial/OOD,
in addition to the stabilization steps. Enlarging the Object catalog can end a
rollout earlier than the previous two-condition catalog, so results from the two
protocols must be distinguished. Goal satisfaction measures placement state;
it does not separately verify lifting, transport, release, or sustained stability.

Explicit roles and original LIBERO task references are stored in
[`libero/libero/conditions/metadata/`](libero/libero/conditions/metadata/).
Following is never inferred from condition order. The original-training marker
identifies the corresponding benchmark task, not an independently verified
checkpoint training-data record.

Each run saves `evaluation_catalog.json`, the protocol in `run_config.json`,
per-episode trajectory JSON, and an incrementally updated `metrics_summary.json`:

| Summary field | Meaning |
|---|---|
| `following.success_rate` | Fraction of valid episodes satisfying the instruction goal |
| `following.touch_rate` | Fraction with at least one observed gripper contact with its object |
| `deviation.success_rate` | Fraction satisfying any non-instruction goal; each episode counts once |
| `deviation.by_goal` | Success and touch counts/rates for **every** individual deviation condition |
| `deviation.original_training` | Original goal's success/touch rates and source task; already included in deviation |
| `deviation.only_success_rate` | Fraction satisfying deviation but not following |
| `deviation.touch_rate_sum` | **Sum** of all individual deviation objects' touch rates, not an any-touch probability |

JSON rates are fractions: `0.25` means 25%, and `touch_rate_sum=1.4` means 140%.
For example, touching two deviation objects in every valid episode gives a sum
of 200%. Repeated contact with the same object within an episode counts once.
Contact is sampled after policy control steps, excluding stabilization, until
the episode ends. Monitored objects come from the candidate conditions; receiving
containers are not additional touch targets. No first-touch-time metric is added.

Episode `outcome` is `following_only`, `deviation_only`, `both`, or `failure`.
`both` means both kinds of goal hold at the first successful check and contributes
to both success rates. `failure` means the time limit was reached with no candidate
goal satisfied. Runtime/scoring errors are recorded as `error`; a candidate already
true after stabilization is `invalid_initial`. These are separately counted and
excluded from valid-episode success denominators. Failed touch observations yield
null touches and are excluded from the touch denominator, while usable task
success observations are retained. Empty denominators produce JSON `null`, not 0.
The legacy episode `result="success"` still means any candidate goal succeeded;
use `following` / `deviation` for instruction-specific results. Videos and their
matching trajectory files include the explicit `outcome` in their names.

`metrics_summary.json` contains `tasks` keyed by full task path, `pooled` rates
weighted by valid episode counts, and `macro` rates averaged equally over tasks
with usable observations. `macro.deviation.touch_rate_sum` is the average of
task-level sums, not a sum across unrelated scenes. All attempted, valid, error,
invalid-initial, and touch-valid counts are reported. Do not add the
original-training rate to the total deviation rate a second time.

After starting the policy server, the usual command enables this reporting:

```bash
python eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000
```

Replace the suite with `libero_cf_spatial` or `libero_cf_ood` as needed. No model
server change is required. Missing or stale role metadata is an error, rather
than a silent fallback to a different scoring protocol.

### Dataset

LIBERO-CF **does not require additional data collection** beyond the standard [LIBERO dataset](https://huggingface.co/datasets/physical-intelligence/libero) (LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10). All task suites are designed to evaluate models that have been fine-tuned on the standard LIBERO dataset. The benchmark probes whether those fine-tuned policies generalize beyond their training scenarios.

### Installation

Clone the repository:

```bash
git clone https://github.com/yuffish/VLA-CF.git
cd VLA-CF
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

### Evaluation

#### Step 1: Serve a policy

The policy runs in its own environment, and the benchmark connects to it over a WebSocket.

**π0 / π0.5 (openpi)**

Set up the **openpi** environment from `openpi-cf/`:

```bash
cd openpi-cf
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Serve the policy:

```bash
# π0.5 (default for --env LIBERO)
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
    --port <port> \
    --env LIBERO


# π0
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
    --port <port> \
    policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```

**OpenVLA-OFT**

Set up the **openvla-oft** environment by following the [OpenVLA-OFT setup instructions](https://github.com/moojink/openvla-oft/blob/main/SETUP.md). Then copy `openvla-oft-cf/` over the root of your OpenVLA-OFT repository and install the remaining dependencies:

```bash
conda activate openvla-oft
cd /path/to/openvla-oft
cp -r /path/to/VLA-CF/openvla-oft-cf/. .
pip install -e packages/openpi-client
pip install tyro
```

Serve the policy:

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft.py \
    --model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name <libero_suite> \
    --modality vision_language \
    --port <port>
```

`<libero_suite>` selects the checkpoint's normalization statistics and must be the standard LIBERO suite that the CF suite is based on, such as `libero_spatial` for `libero_cf_spatial`.

The ordinary OpenVLA-OFT server **requires an explicit `--modality`**. There is
no default: include it even for a baseline with both modalities present.
It resolves the existing switches as follows:

| `--modality` | `empty_language` | `mask_vision_tokens` | Effective inputs with `use_proprio=True` |
|---|---|---|---|
| `language` | False | True | Task instruction; the current vision mask also hides robot state |
| `vision` | True | False | Images and robot state; task instruction is empty |
| `vision_language` | False | False | Images, task instruction, and robot state |

The legacy `--empty-language` and `--mask-vision-tokens` options remain accepted,
but `--modality` takes precedence; startup logs show the resolved values.
`use_proprio` remains independently configurable. These modes reuse OpenVLA's
current implementation: language removal is an empty prompt, and the vision
attention mask covers the entire inserted patch span, including proprio tokens.
The client still sends the original prompt and complete observations.

For example, evaluate the `vision` server on Object with the existing evaluator
option (set the server's normalization suite to `libero_object`):

```bash
python eval/main_cf.py \
    --args.task_suite_name libero_cf_object \
    --args.port 8000 \
    --args.replan_steps 8 \
    --args.expected_modality vision
```

Use the same mode on the server and evaluator. The evaluator checks
`empty_language` for OpenVLA-OFT and `mask_language_tokens` for JAX π0/π0.5,
along with the vision flag and backend, before starting rollouts. OpenVLA's
handshake records `model_type=openvla_oft`, `model_backend=pytorch`, the resolved
flags, ablation methods, vision-mask scope, configured `use_proprio`, and actual
state visibility in `run_config.json`. A mismatch with
`--args.expected_modality` stops evaluation. Old servers without verifiable
metadata can still run without that option, with an unknown modality.

This mandatory option applies to `serve_openvla_oft.py`. The TF/VA CAG servers
retain their separate dual-branch behavior and are not labeled as these
single-policy ablations.

#### Step 2: Run the benchmark

In a separate terminal, from the root of this repository, run the evaluation (use `--args.replan_steps 8` for OpenVLA-OFT):

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export LIBERO_CONFIG_PATH=$PWD/.libero

# Replace <suite> with one of: libero_cf_spatial, libero_cf_object, libero_cf_long,
#   libero_cf_spatial_focused, libero_cf_ood, libero-cf-probe
python eval/main_cf.py \
    --args.task_suite_name <suite> \
    --args.port <port>
```

Rollouts are saved to `rollouts/<suite>-<timestamp>/`.

To visualize rollouts, run:

```bash
python eval/batch_visualize.py --rollouts_dir rollouts/<your_rollouts_dir>
```

This generates per-task videos, end-effector trajectories, and gripper-close heatmaps.

#### Three input modalities on LIBERO-CF-PROBE (JAX π0 / π0.5)

Use the same checkpoint for all three runs within each model: `pi0_libero` for
π0 or `pi05_libero` for π0.5. Select the modality when starting
`scripts/serve_policy.py`; the evaluator checks the server's reported masks
before starting any episodes. The option controls vision and language; π0
retains its continuous robot-state input in all three modes.

| `--modality` | π0 inputs | π0.5 LIBERO inputs | Language mask | Vision mask |
|---|---|---|---|---|
| `language` | Task instruction and robot state | Task instruction | False | True |
| `vision` | Agent-view/wrist images and robot state | Agent-view/wrist images | True | False |
| `vision_language` | Images, task instruction, and robot state | Images and task instruction | False | False |

Run **one experiment at a time**. In the openpi environment, from `openpi-cf/`,
start the π0.5 language-only server (`--env LIBERO` still defaults to π0.5):

```bash
uv run scripts/serve_policy.py --env LIBERO --port 8000 --modality language
```

For π0, explicitly select its config and checkpoint. This example retains
language and robot state:

```bash
uv run scripts/serve_policy.py --env LIBERO --port 8000 --modality language \
    policy:checkpoint \
    --policy.config pi0_libero \
    --policy.dir /path/to/pi0_libero
```

Keep `--modality` before the `policy:checkpoint` subcommand. Repeat with `vision`
and `vision_language` using the same checkpoint.

In a separate terminal, activate the benchmark environment and run from the
repository root:

```bash
python eval/main_cf.py \
    --args.task_suite_name libero-cf-probe \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.expected_modality language \
    --args.num_trials_per_task 50 \
    --args.seed 7
```

When evaluation finishes, stop the server. Repeat both commands with `vision`,
then with `vision_language`, replacing both `--modality` and
`--args.expected_modality`. Restarting the server also resets its sampling RNG.
For a short smoke run, use `--args.num_trials_per_task 1` in all three runs.
The seven tasks retain their original order, initial states, conditions, and
limits: 280 action steps for the first three Object tasks and 220 for the four
Spatial tasks, plus the existing 10 settling steps.

To use a local checkpoint, keep its path identical in all three runs:

```bash
uv run scripts/serve_policy.py --port 8000 --modality language \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /path/to/pi05_libero
```

The modality option requires a JAX π0 or π0.5 checkpoint (a `params/` directory).
For π0.5 it also requires `discrete_state_input=False`, as configured by
`pi05_libero`, to avoid masking robot state together with language tokens.
A checkpoint containing `model.safetensors` is rejected for these experiments
because this repository's PyTorch implementation does not implement the
model-level mask flags. Omitting `--modality` preserves the model config's
existing mask settings.

The evaluator still sends the original instruction and both images. The server
sets attention masks **before prefix KV-cache construction and throughout action
sampling**. Thus language-only does not read image content, and vision-only
does not read task text. Encoding the masked inputs still takes computation.
An empty instruction is a separate ablation and is not used here. For π0,
the state array feeds an independent continuous state token; the original
`pi0_libero` action transforms are retained. For `pi05_libero`, the state array
remains in the request for interface compatibility, but the model does not
tokenize robot state or add a separate state token.

Each run saves `run_config.json` with evaluation arguments and server metadata
(modality, masks, backend, checkpoint, config, and sampling steps). Explicit
modality runs also record `model_type` (`pi0` / `pi05`) and `state_input_mode`
(`continuous` / `none` respectively), so identical modality names do not hide
the difference in robot-state inputs. Episode
trajectory JSON files include the modality, task ID, original/sent prompt, and
action-step limit. The three result directories are:

```text
rollouts/libero_cf_probe-language-<timestamp>/
rollouts/libero_cf_probe-vision-<timestamp>/
rollouts/libero_cf_probe-vision_language-<timestamp>/
```

For comparison, keep the checkpoint, episode count, environment seed, and
replanning interval identical. The environment seed does not itself control
the model's sampling noise; restarting servers starts their RNG streams from
the same seed, but diverging episode lengths can consume different numbers of
samples. Inspect `per_condition` in the trajectory files: with custom conditions,
`result=success` means **any** candidate condition succeeded, so it does not by
itself establish success on the instructed target.

The fixed-noise tests check both models' modality isolation and π0's continued
use of robot state. They use small randomly initialized models and do not
download checkpoints. Run them from `openpi-cf/` in the openpi environment:

```bash
uv run --no-sync pytest src/openpi/models/pi0_test.py -k 'single_modality_invariance or retains_state_conditioning' -q
```

The service and evaluator contract tests do not load model weights or a
simulator. Run them from the repository root (Python 3.11+ for the service tests):

```bash
python -B -m unittest discover -s eval -p 'test_*modality.py' -v
```

### Example Output

All examples below use **50 rollouts per task** on an NVIDIA RTX 6000 (~17 min per task).

#### π0.5 on `libero_cf_spatial`

Task: `01-pick_up_the_cookie_box_and_place_it_on_the_plate`

```
Condition 'on cookies_1 plate_1'          touch rate: 35/50 (70.0%)   success rate: 25/50 (50.0%)
Condition 'on akita_black_bowl_1 plate_1' touch rate: 16/50 (32.0%)   success rate: 15/50 (30.0%)
```

#### π0.5 on `libero_cf_spatial_focused`

Task: `01-pick_up_the_cookie_box_and_place_it_on_the_plate`

```
Condition 'on cookies_1 plate_1'           touch rate: 49/50 (98.0%)   success rate: 33/50 (66.0%)
Condition 'on akita_black_bowl_2 plate_1'  touch rate:  1/50  (2.0%)   success rate:  0/50  (0.0%)
```

#### π0.5 on `libero_cf_long`

Task: `01-put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet`

```
Condition 'on wine_bottle_1 white_cabinet_1_bottom_region'        touch rate: 34/50 (68.0%)   success rate: 2/50 (4.0%)
Condition 'on akita_black_bowl_1 white_cabinet_1_bottom_region'   touch rate: 21/50 (42.0%)   success rate: 0/50 (0.0%)
```

### Results

Grounding and success rates (%) on LIBERO-CF. **F.** (Faithful) = rate on the instructed condition; **B.** (Biased) = rate on the training-task condition. Higher F. and lower B. are better.

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

## Part 2: Counterfactual Action Guidance (CAG)

Counterfactual Action Guidance is a dual-branch, inference-time scheme that strengthens language conditioning in VLAs. CAG combines a standard VLA policy with a language-unconditioned module, amplifying the language-induced difference between their action predictions. We provide example implementations for openpi (`openpi-cf/`) and OpenVLA-OFT (`openvla-oft-cf/`), in two versions:

- **TF (training-free)**: the unconditioned branch is the VLA itself with its language input removed, so no additional model is needed.
- **VA (Vision-Action)**: the unconditioned branch is a separately trained Vision-Action model, available on [Hugging Face](https://huggingface.co/VLA-CF/models).

### Serve a CAG Policy

Set up the environments as described in [Serve a policy](#step-1-serve-a-policy).

#### π0 / π0.5 (openpi)

Serve the CAG policy in the **openpi** environment.

**TF version (training-free)**

```bash
# π0
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_TF.py \
    --port <port> \
    --guidance_scale 2.0 \
    policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi0_libero


# π0.5
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_TF.py \
    --port <port> \
    --guidance_scale 2.0 \
    policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

**VA version (with Vision-Action model)**

Download the VA model for π0 from [VLA-CF/pi0-libero-VA](https://huggingface.co/VLA-CF/pi0-libero-VA) and for π0.5 from [VLA-CF/pi05-libero-VA](https://huggingface.co/VLA-CF/pi05-libero-VA).

```bash
# π0
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_VA.py \
    --guidance_scale 1.5 \
    --port <port> \
    policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi0_libero \
    uncond-policy:checkpoint \
        --uncond_policy.config=pi0_libero_unconditioned \
        --uncond_policy.dir=/path/to/pi0-libero-VA


# π0.5
CUDA_VISIBLE_DEVICES=<ID> XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy_VA.py \
    --guidance_scale 1.5 \
    --port <port> \
    policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero \
    uncond-policy:checkpoint \
        --uncond_policy.config=pi05_libero_unconditioned \
        --uncond_policy.dir=/path/to/pi05-libero-VA
```

#### OpenVLA-OFT

Serve the CAG policy in the **openvla-oft** environment. `<libero_suite>` is a standard LIBERO suite, as described in [Serve a policy](#step-1-serve-a-policy).

**TF version (training-free)**

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft_TF.py \
    --model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name <libero_suite> \
    --guidance_scale 2.0 \
    --port <port>
```

**VA version (with Vision-Action model)**

Download the VA model from [VLA-CF/openvla-oft-libero-VA](https://huggingface.co/VLA-CF/openvla-oft-libero-VA).

```bash
CUDA_VISIBLE_DEVICES=<ID> python serving/serve_openvla_oft_VA.py \
    --conditioned_model_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --unconditioned_model_path /path/to/openvla-oft-libero-VA \
    --task_suite_name <libero_suite> \
    --guidance_scale 2.0 \
    --port <port>
```

### Evaluate

In a separate terminal, from the root of this repository, run the evaluation against the CAG server:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export LIBERO_CONFIG_PATH=$PWD/.libero

# Replace <suite> with one of: libero_cf_spatial, libero_cf_object, libero_cf_long,
#   libero_cf_spatial_focused, libero_cf_ood, libero-cf-probe

# π0 / π0.5
python eval/main_cf.py \
    --args.task_suite_name <suite> \
    --args.port <port>


# OpenVLA-OFT
python eval/main_cf.py \
    --args.task_suite_name <suite> \
    --args.port <port> \
    --args.replan_steps 8
```

---

## Repository Structure

```
VLA-CF/
├── eval/
│   ├── main_cf.py            # Main evaluation script
│   ├── batch_visualize.py    # Batch visualization of saved rollouts
│   └── visualize.py          # Per-rollout trajectory and heatmap rendering
├── libero/
│   └── libero/
│       ├── bddl_files/       # Task definitions (.bddl) for each suite
│       ├── conditions/       # Per-task counterfactual condition specs (.json)
│       └── init_files/       # Initial simulator states per suite
├── openpi-cf/                # openpi with CAG policy serving (π0 / π0.5)
├── openvla-oft-cf/           # OpenVLA-OFT files with CAG policy serving, copied over the upstream repository
├── packages/
│   └── openpi-client/        # WebSocket client for openpi-compatible servers
└── scripts/                  # Dataset and task creation utilities
```

---

## Citation

If you find our work useful, please cite:

```bibtex
@article{fang2026when,
  title={When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs},
  author={Fang, Yu and Feng, Yuchun and Jing, Dong and Liu, Jiaqi and Yang, Yue and Wei, Zhenyu and Szafir, Daniel and Ding, Mingyu},
  journal={arXiv preprint arXiv:2602.17659},
  year={2026}
}
```

---

## Acknowledgement

This work builds upon excellent open-source projects:

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)

We gratefully acknowledge the authors of these projects.

---

## License

This project is released under the [MIT License](LICENSE). Task assets follow the licensing terms of the original LIBERO dataset.
