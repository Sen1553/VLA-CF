import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tyro
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from openpi_client import base_policy as _base_policy
from openvla_oft_policy import OpenVLAOFTPolicy
from websocket_policy_server import WebsocketPolicyServer


class CounterfactualActionGuidedOpenVLAOFTPolicy(_base_policy.BasePolicy):
    """Wrap an OpenVLA-OFT policy with counterfactual action guidance.

    Runs two forward passes (conditional + empty language) and fuses actions with CAG.
    """

    def __init__(self, base_policy: OpenVLAOFTPolicy, guidance_scale: float = 1.0) -> None:
        self._base_policy = base_policy
        self._guidance_scale = guidance_scale
        self._metadata: Dict[str, Any] = dict(getattr(base_policy, "metadata", {}))
        self._metadata["cag_guidance_scale"] = guidance_scale

        self._use_l1_regression = base_policy.cfg.use_l1_regression
        if self._use_l1_regression:
            logging.info("CAG: using action-space guidance (L1 regression outputs).")
        else:
            logging.info("CAG: using logit-space guidance (discrete token logits).")

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def _run_pass(self, obs: Dict[str, Any], *, empty_language: bool) -> Dict[str, Any]:
        obs_for_pass = dict(obs)
        if empty_language:
            obs_for_pass["prompt"] = ""
        return self._base_policy.infer_with_logits(obs_for_pass)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        cond_outputs = self._run_pass(obs, empty_language=False)
        uncond_outputs = self._run_pass(obs, empty_language=True)

        cond_logits = cond_outputs.get("action_logits")
        uncond_logits = uncond_outputs.get("action_logits")
        outputs = dict(cond_outputs)

        if self._use_l1_regression:
            cond_actions = np.asarray(cond_outputs["actions"], dtype=np.float32)
            uncond_actions = np.asarray(uncond_outputs["actions"], dtype=np.float32)
            guided_actions = uncond_actions + self._guidance_scale * (cond_actions - uncond_actions)
            outputs["actions"] = guided_actions
            outputs["action_logits"] = None
        else:
            guided_logits = np.asarray(uncond_logits, dtype=np.float32) + self._guidance_scale * (
                np.asarray(cond_logits, dtype=np.float32) - np.asarray(uncond_logits, dtype=np.float32)
            )
            guided_actions = self._base_policy.decode_logits_to_actions(guided_logits)
            guided_actions = self._base_policy._postprocess(guided_actions)
            outputs["actions"] = guided_actions
            outputs["action_logits"] = guided_logits

        return outputs


@dataclass
class Args:
    model_path: str
    port: int = 8010
    device: str = "cuda"
    task_suite_name: str = "libero_spatial"
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    use_proprio: bool = True
    num_images_in_input: int = 2
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    guidance_scale: float = 1.0
    use_l1_regression: bool = True
    mask_vision_tokens: bool = False


def main(args: Args):
    print(f"Serving OpenVLA-OFT policy with CAG guidance scale: {args.guidance_scale}")
    base_policy = OpenVLAOFTPolicy(
        args.model_path,
        device=args.device,
        empty_language=False,
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
    policy = CounterfactualActionGuidedOpenVLAOFTPolicy(base_policy, guidance_scale=args.guidance_scale)
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
