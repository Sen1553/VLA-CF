import dataclasses
import enum
import logging
import socket
from typing import Literal

import tyro

from openpi.models import pi0_config as _pi0_config
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
    default_prompt: str | None = None

    # If True, overwrite any existing prompt with the provided default.
    overwrite: bool = False

    # Probability of injecting the default prompt. If 0, this transform is a no-op.
    probability: float = 0.0

    # Explicit input ablation for JAX pi05_libero. None preserves the checkpoint config.
    modality: Literal["language", "vision", "vision_language"] | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

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


def _create_checkpoint_policy(
    checkpoint: Checkpoint,
    *,
    default_prompt: str | None = None,
    overwrite: bool = False,
    probability: float = 0.0,
    modality: Literal["language", "vision", "vision_language"] | None = None,
) -> _policy.Policy:
    config = _config.get_config(checkpoint.config)
    metadata = {
        **(config.policy_metadata or {}),
        "training_config": checkpoint.config,
        "checkpoint": checkpoint.dir,
    }
    if modality is not None:
        if modality not in ("language", "vision", "vision_language"):
            raise ValueError(f"Unsupported input modality: {modality}")
        if not isinstance(config.model, _pi0_config.Pi0Config) or not config.model.pi05:
            raise ValueError("--modality requires a pi0.5 config, such as pi05_libero.")
        if config.model.discrete_state_input:
            raise ValueError(
                "--modality requires discrete_state_input=False (as in pi05_libero), "
                "so masking language does not also remove robot state tokens."
            )
        # Configure before Policy creates its frozen JIT wrapper.
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                mask_language_tokens=modality == "vision",
                mask_vision_tokens=modality == "language",
            ),
        )
        metadata.update(
            input_modality=modality,
            mask_language_tokens=config.model.mask_language_tokens,
            mask_vision_tokens=config.model.mask_vision_tokens,
            sampling_num_steps=10,
        )
    config = dataclasses.replace(config, policy_metadata=metadata)
    return _policy_config.create_trained_policy(
        config,
        checkpoint.dir,
        default_prompt=default_prompt,
        overwrite=overwrite,
        probability=probability,
        sample_kwargs={"num_steps": 10} if modality is not None else None,
        require_jax=modality is not None,
    )


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    overwrite: bool = False,
    probability: float = 0.0,
    modality: Literal["language", "vision", "vision_language"] | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _create_checkpoint_policy(
            checkpoint,
            default_prompt=default_prompt,
            overwrite=overwrite,
            probability=probability,
            modality=modality,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _create_checkpoint_policy(
                args.policy,
                default_prompt=args.default_prompt,
                overwrite=args.overwrite,
                probability=args.probability,
                modality=args.modality,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                overwrite=args.overwrite,
                probability=args.probability,
                modality=args.modality,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    logging.info("Policy metadata: %s", policy_metadata)

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
