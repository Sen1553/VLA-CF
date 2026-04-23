# scripts/serve_openvla.py
import logging
from dataclasses import dataclass

import tyro
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from websocket_policy_server import WebsocketPolicyServer

from openvla_oft_policy import OpenVLAOFTPolicy


@dataclass
class Args:
    model_path: str
    port: int = 8010
    device: str = "cuda"
    empty_language: bool = False
    task_suite_name: str = "libero_spatial"
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    use_proprio: bool = True
    num_images_in_input: int = 2
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_l1_regression: bool = True
    mask_vision_tokens: bool = False


def main(args: Args):
    print(f"Serving OpenVLA-OFT policy with empty language: {args.empty_language}")
    print(f"Serving OpenVLA-OFT policy with mask vision tokens: {args.mask_vision_tokens}")
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
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=getattr(policy, "metadata", {}),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))