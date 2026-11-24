# scripts/serve_openvla.py
import logging
from dataclasses import dataclass, replace
from typing import Literal

import tyro
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from websocket_policy_server import WebsocketPolicyServer

from openvla_oft_policy import OpenVLAOFTPolicy


@dataclass
class Args:
    model_path: str
    # Required vision/language ablation. Takes precedence over the legacy flags below.
    modality: Literal["language", "vision", "vision_language"]
    port: int = 8010
    device: str = "cuda"
    # Legacy flag, resolved from --modality before creating the policy.
    empty_language: bool = False
    task_suite_name: str = "libero_spatial"
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    use_proprio: bool = True
    num_images_in_input: int = 2
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_l1_regression: bool = True
    # Legacy flag, resolved from --modality before creating the policy.
    mask_vision_tokens: bool = False


def main(args: Args):
    if args.modality not in ("language", "vision", "vision_language"):
        raise ValueError(f"Unsupported input modality: {args.modality!r}")
    args = replace(
        args,
        empty_language=args.modality == "vision",
        mask_vision_tokens=args.modality == "language",
    )
    logging.info(
        "Serving OpenVLA-OFT with modality=%s, empty_language=%s, mask_vision_tokens=%s, use_proprio=%s",
        args.modality, args.empty_language, args.mask_vision_tokens, args.use_proprio,
    )
    policy = OpenVLAOFTPolicy(
        args.model_path,
        device=args.device,
        empty_language=args.empty_language,
        task_suite_name=args.task_suite_name,
        num_open_loop_steps=args.num_open_loop_steps,
        use_proprio=args.use_proprio,
        num_images_in_input=args.num_images_in_input,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_l1_regression=args.use_l1_regression,
        mask_vision_tokens=args.mask_vision_tokens,
    )
    # Describe the existing ablations without changing the model or the shared
    # policy metadata (which is also inherited by the dual-branch CAG servers).
    empty_language = policy.empty_language
    mask_vision_tokens = policy.cfg.mask_vision_tokens
    use_proprio = policy.cfg.use_proprio
    state_visible = use_proprio and not mask_vision_tokens
    metadata = {
        **getattr(policy, "metadata", {}),
        "model_type": "openvla_oft",
        "model_backend": "pytorch",
        "input_modality": args.modality,
        "empty_language": empty_language,
        "mask_vision_tokens": mask_vision_tokens,
        "language_ablation_method": "empty_prompt" if empty_language else "none",
        "vision_ablation_method": "attention_mask" if mask_vision_tokens else "none",
        # The current model masks the entire inserted span, including proprio.
        "vision_mask_scope": "inserted_patch_span" if mask_vision_tokens else "none",
        "use_proprio": use_proprio,
        "state_input_visible": state_visible,
        "state_input_mode": "continuous" if state_visible else "none",
        "checkpoint": args.model_path,
        "num_images_in_input": policy.cfg.num_images_in_input,
        "task_suite_name": policy.cfg.task_suite_name,
    }
    logging.info("Policy metadata: %s", metadata)
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
