import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.gemma as _gemma
import openpi.models.pi0_config as _pi0_config
import openpi.models.siglip as _siglip
from openpi.shared import nnx_utils


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


@pytest.fixture
def tiny_pi05(monkeypatch):
    """Keep real attention and sampling, without pretrained weights or large encoders."""
    monkeypatch.setattr(_gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    siglip_module = _siglip.Module

    def tiny_siglip(*args, **kwargs):
        kwargs.update(
            width=16,
            depth=1,
            mlp_dim=32,
            num_heads=2,
            patch_size=(112, 112),
            head_zeroinit=False,
        )
        return siglip_module(*args, **kwargs)

    monkeypatch.setattr(_siglip, "Module", tiny_siglip)
    return _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=7,
        action_horizon=3,
        max_token_len=4,
    )


def _enable_random_action_expert_gates(model):
    # Pi05 initializes adaRMS residual gates to zero. Such an untrained expert
    # ignores *all* prefix inputs, making an invariance test pass vacuously.
    state = nnx.state(model, nnx.Param)
    params = state.to_pure_dict()
    changed = 0

    def enable_gates(node, path=()):
        nonlocal changed
        for name, value in node.items():
            if isinstance(value, dict):
                enable_gates(value, (*path, name))
            elif name == "bias" and any(part in {"pre_attention_norm_1", "pre_ffw_norm_1"} for part in path):
                # Dense output packs scale, shift and gate along the last axis;
                # the leading axis contains the scanned Transformer layers.
                width = value.shape[-1] // 3
                node[name] = value.at[..., 2 * width :].set(1.0)
                changed += 1

    enable_gates(params)
    assert changed == 2, "Expected attention and feed-forward adaRMS gates"
    state.replace_by_pure_dict(params)
    nnx.update(model, state)


@pytest.mark.parametrize("modality", ["language", "vision"])
def test_pi05_single_modality_invariance(tiny_pi05, modality):
    """Removed inputs cannot leak through prefix prefill or denoising attention."""
    config = dataclasses.replace(
        tiny_pi05,
        mask_language_tokens=modality == "vision",
        mask_vision_tokens=modality == "language",
    )
    model = config.create(jax.random.key(0))
    _enable_random_action_expert_gates(model)

    observation = config.fake_obs()
    observation = observation.replace(tokenized_prompt=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32))
    if modality == "language":
        changed_observation = observation.replace(
            images={
                name: jax.random.uniform(jax.random.key(10 + index), image.shape, minval=-1.0, maxval=1.0)
                for index, (name, image) in enumerate(observation.images.items())
            }
        )
    else:
        changed_observation = observation.replace(tokenized_prompt=jnp.array([[17, 29, 41, 53]], dtype=jnp.int32))

    # Positive control: exactly the same parameters, now with both modalities
    # visible, must respond to the perturbation. Clone before freezing with JIT.
    unmasked_model = nnx.clone(model)
    unmasked_model.mask_language_tokens = False
    unmasked_model.mask_vision_tokens = False
    sample = nnx_utils.module_jit(model.sample_actions)
    sample_unmasked = nnx_utils.module_jit(unmasked_model.sample_actions)
    rng = jax.random.key(1)
    noise = jax.random.normal(rng, (1, config.action_horizon, config.action_dim))

    actions = sample(rng, observation, noise=noise, num_steps=3)
    changed_actions = sample(rng, changed_observation, noise=noise, num_steps=3)
    assert actions.shape == (1, config.action_horizon, config.action_dim)
    assert np.isfinite(actions).all()
    np.testing.assert_allclose(actions, changed_actions, rtol=1e-6, atol=1e-6)

    unmasked_actions = sample_unmasked(rng, observation, noise=noise, num_steps=3)
    unmasked_changed_actions = sample_unmasked(rng, changed_observation, noise=noise, num_steps=3)
    assert np.isfinite(unmasked_actions).all()
    assert np.isfinite(unmasked_changed_actions).all()
    assert not np.allclose(unmasked_actions, unmasked_changed_actions, rtol=1e-5, atol=1e-5)
