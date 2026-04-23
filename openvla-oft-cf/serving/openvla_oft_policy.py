import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import math
import random
import os
from enum import Enum

from openpi_client import base_policy as _base_policy
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_vla
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["observation/image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["observation/wrist_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclass
class _ServingConfig:
    """Lightweight config mirroring the LIBERO eval defaults we need for inference."""

    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    unnorm_key: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_spatial"
    seed: int = 7
    mask_vision_tokens: bool = False


class OpenVLAOFTPolicy(_base_policy.BasePolicy):
    """
    Wrap OpenVLA with the same preprocessing / postprocessing used in LIBERO eval,
    but expose it over the websocket policy interface.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        empty_language: bool = False,
        task_suite_name: str = "libero_spatial",
        num_open_loop_steps: Optional[int] = None,
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        center_crop: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_l1_regression: bool = True,
        mask_vision_tokens: bool = False,
    ):
        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.empty_language = empty_language

        # Build a minimal config that matches the LIBERO evaluation defaults
        self.cfg = _ServingConfig(
            pretrained_checkpoint=model_path,
            task_suite_name=task_suite_name,
            num_open_loop_steps=num_open_loop_steps or NUM_ACTIONS_CHUNK,
            use_proprio=use_proprio,
            num_images_in_input=num_images_in_input,
            center_crop=center_crop,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_l1_regression=use_l1_regression,
            mask_vision_tokens=mask_vision_tokens,
        )
        self.validate_config(self.cfg)

        # Load model + helpers
        (
            self.model,
            self.action_head,
            self.proprio_projector,
            self.noisy_action_projector,
            self.processor,
        ) = self._initialize_model_components()
        self.model.eval()

        # Cache resize size and open-loop buffer
        self.resize_size = get_image_resize_size(self.cfg)
        self.set_seed_everywhere(self.cfg.seed)

        # Metadata for clients
        self._metadata: Dict[str, Any] = {
            "policy_name": "OpenVLA",
            "expects": {
                # Primary scene image (agentview); we also accept already-processed `full_image`
                "full_image": {"keys": ["observation/full_image", "observation/agentview_image"], "shape": [-1, -1, 3], "dtype": "uint8"},
                # Optional wrist camera; only used if provided
                "wrist_image": {"keys": ["observation/wrist_image", "observation/robot0_eye_in_hand_image"], "optional": True},
                # Proprio state (eef pos + quat->axisangle + gripper)
                "state": {"keys": ["observation/state"], "optional": not self.cfg.use_proprio, "shape": [8], "dtype": "float32"},
                # Task description
                "prompt": {"type": "str"},
            },
            "produces": {
                # Single-step action (policy caches an open-loop chunk internally)
                "actions": {"shape": [7], "dtype": "float32"},
            },
        }


    def validate_config(self, cfg: _ServingConfig) -> None:
        """Validate configuration parameters."""
        assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

        if "image_aug" in str(cfg.pretrained_checkpoint):
            assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

        # Validate task suite
        assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


    def set_seed_everywhere(self, seed: int) -> None:
        """
        Set random seed for all random number generators for reproducibility.

        Args:
            seed: The random seed to use
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    # --------------------------------------------------------------------- #
    # Initialization helpers
    # --------------------------------------------------------------------- #
    def _initialize_model_components(self):
        model = get_vla(self.cfg)
        # Ablation: mask vision tokens (implemented in Prismatic model attention mask construction).
        mask_vision_tokens = getattr(self.cfg, "mask_vision_tokens", False)
        if mask_vision_tokens is True:
            setattr(model, "mask_vision_tokens", True)
        elif mask_vision_tokens is False:
            setattr(model, "mask_vision_tokens", False)
        else:
            raise TypeError(
                f"Expected cfg.mask_vision_tokens to be a bool, got {type(mask_vision_tokens)}: {mask_vision_tokens}"
            )

        # Optional heads / projectors matching LIBERO evaluation defaults
        action_head = None
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            action_head = get_action_head(self.cfg, model.llm_dim)

        proprio_projector = None
        if self.cfg.use_proprio:
            proprio_projector = get_proprio_projector(self.cfg, model.llm_dim, proprio_dim=8)

        noisy_action_projector = None
        if self.cfg.use_diffusion:
            noisy_action_projector = get_noisy_action_projector(self.cfg, model.llm_dim)

        processor = get_processor(self.cfg)

        self._set_unnorm_key(model)
        # Move model to the requested device if not already there
        model.to(self.device)

        return model, action_head, proprio_projector, noisy_action_projector, processor

    def _set_unnorm_key(self, model) -> None:
        """Mirror the LIBERO eval logic for choosing an unnorm key."""
        norm_stats = getattr(model, "norm_stats", None)
        if not norm_stats or not isinstance(norm_stats, dict) or len(norm_stats) == 0:
            logging.warning("No norm_stats found on model; skipping unnorm_key selection.")
            return

        unnorm_key = self.cfg.task_suite_name
        if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

        if unnorm_key not in norm_stats:
            logging.warning(
                "Action un-norm key %s not found in norm_stats; falling back to first available key.",
                unnorm_key,
            )
            unnorm_key = next(iter(norm_stats.keys()))

        self.cfg.unnorm_key = unnorm_key

    # --------------------------------------------------------------------- #
    # Pre/post-processing mirroring run_libero_eval.py
    # --------------------------------------------------------------------- #

    def _prepare_observation(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        # Primary + wrist images (rotate raw LIBERO frames to match training)
        img = obs["observation/image"]
        wrist_img = obs["observation/wrist_image"]
        observation = {
            "full_image": img,
            "wrist_image": wrist_img,
            "state": obs["observation/state"],
        }

        return observation

    def _postprocess(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return action

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default inference path used by the websocket server.
        """
        result = self._infer_with_optionally_returned_logits(obs, return_logits=False)
        return {"actions": result["actions"]}

    def infer_with_logits(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference path that also returns action logits (when available) for CAG.
        """
        return self._infer_with_optionally_returned_logits(obs, return_logits=True)

    def _infer_with_optionally_returned_logits(self, obs: Dict[str, Any], return_logits: bool) -> Dict[str, Any]:
        observation = self._prepare_observation(obs)
        task_description = obs["prompt"]
        if self.empty_language:
            task_description = ""

        result = get_action(
            self.cfg,
            self.model,
            observation,
            task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
            return_logits=return_logits,
        )

        if return_logits:
            actions, action_logits = result
        else:
            actions = result
            action_logits = None

        outputs = {"actions": self._postprocess(actions)}
        if return_logits:
            outputs["action_logits"] = action_logits
        return outputs

    def decode_logits_to_actions(self, action_logits: np.ndarray) -> np.ndarray:
        """
        Convert action token logits into unnormalized continuous actions using model helpers.
        """
        action_logits_tensor = torch.as_tensor(action_logits, device=self.device)
        normalized_actions = self.model._decode_action_logits(action_logits_tensor)
        return self.model._unnormalize_actions(normalized_actions, self.cfg.unnorm_key)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata