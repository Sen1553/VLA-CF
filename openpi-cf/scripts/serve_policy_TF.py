import dataclasses
import enum
import logging
import socket
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str = ""

    # If True, the unconditional pass uses an empty prompt (re-run transforms with an empty string).
    # If False, the unconditional pass masks language tokens after tokenization (current default).
    uncond_empty_prompt: bool = False

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Guidance scale to apply when combining conditional/unconditional passes.
    guidance_scale: float | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


class CounterfactualActionGuidedPolicy:
    """Run two forward passes (conditional + language-masked) and fuse with CAG."""

    def __init__(
        self,
        base_policy: _policy.Policy,
        *,
        guidance_scale: float | None = None,
        uncond_empty_prompt: bool = False,
    ):
        self._base_policy = base_policy
        self._guidance_scale = guidance_scale if guidance_scale is not None else getattr(base_policy._model, "guidance_scale", 1.0)
        self._uncond_empty_prompt = uncond_empty_prompt
        self._metadata: dict[str, Any] = dict(base_policy.metadata)
        self._metadata["cag_guidance_scale"] = self._guidance_scale
        self._metadata["cag_uncond_empty_prompt"] = uncond_empty_prompt

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _mask_language_tokens(self, inputs: dict) -> dict:
        masked = dict(inputs)
        if masked.get("tokenized_prompt") is not None:
            masked["tokenized_prompt"] = jnp.zeros_like(masked["tokenized_prompt"])
        if masked.get("tokenized_prompt_mask") is not None:
            masked["tokenized_prompt_mask"] = jnp.zeros_like(masked["tokenized_prompt_mask"])
        return masked

    def _unbatch_to_numpy(self, value):
        return np.asarray(value[0, ...])

    def _prepare_inputs(self, obs: dict, *, empty_prompt: bool = False):
        # Copy so transforms cannot mutate caller state.
        inputs = jax.tree.map(lambda x: x, obs)
        if empty_prompt:
            # Ensure prompt key exists and is empty before running transforms.
            inputs = dict(inputs)
            inputs["prompt"] = ""
        inputs = self._base_policy._input_transform(inputs)
        return inputs

    def _convert_inputs(self, inputs: dict):
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    def _prepare_noise(self, provided_noise, *, noise_rng, action_shape: tuple[int, int]):
        if provided_noise is None:
            noise = jax.random.normal(noise_rng, (1, *action_shape))
        else:
            noise = jnp.asarray(provided_noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
        return noise

    def infer(self, obs: dict, *, noise: np.ndarray | None = None):  # type: ignore[misc]
        inputs = self._prepare_inputs(obs, empty_prompt=False)
        if self._uncond_empty_prompt:
            masked_inputs = self._prepare_inputs(obs, empty_prompt=True)
        else:
            masked_inputs = self._mask_language_tokens(inputs)
        sample_kwargs = dict(self._base_policy._sample_kwargs)
        model = self._base_policy._model

        inputs = self._convert_inputs(inputs)
        masked_inputs = self._convert_inputs(masked_inputs)
        rng_master = getattr(self._base_policy, "_rng", jax.random.key(0))
        self._base_policy._rng, noise_rng, cond_rng, uncond_rng = jax.random.split(rng_master, 4)
        noise_tensor = self._prepare_noise(noise, noise_rng=noise_rng, action_shape=(model.action_horizon, model.action_dim))

        sample_kwargs["noise"] = noise_tensor

        observation = _model.Observation.from_dict(inputs)
        masked_observation = _model.Observation.from_dict(masked_inputs)

        start_time = time.monotonic()
        cond_actions = self._base_policy._sample_actions(cond_rng, observation, **sample_kwargs)
        uncond_actions = self._base_policy._sample_actions(uncond_rng, masked_observation, **sample_kwargs)

        cond_actions_np = self._unbatch_to_numpy(cond_actions)
        uncond_actions_np = self._unbatch_to_numpy(uncond_actions)
        guided_actions = uncond_actions_np + self._guidance_scale * (cond_actions_np - uncond_actions_np)

        outputs = {
            "state": self._unbatch_to_numpy(inputs["state"]),
            "actions": guided_actions,
        }
        outputs = self._base_policy._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - start_time) * 1000}
        return outputs


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    logging.info("Policy args: %s", args.policy)
    logging.info("Config: %s", _config.get_config(args.policy.config))
    match args.policy:
        case Checkpoint():
            base_policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir
            )
        case Default():
            base_policy = _policy_config.create_trained_policy(
                _config.get_config(DEFAULT_CHECKPOINT[args.env].config),
                DEFAULT_CHECKPOINT[args.env].dir,
            )

    return CounterfactualActionGuidedPolicy(
        base_policy,
        guidance_scale=args.guidance_scale,
        uncond_empty_prompt=args.uncond_empty_prompt,
    )


def main(args: Args) -> None:
    logging.info("Args: %s", args)
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
