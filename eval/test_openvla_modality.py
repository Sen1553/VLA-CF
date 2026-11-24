"""Ordinary OpenVLA server contracts without Torch, model weights, or a socket."""

import ast
import contextlib
import copy
import dataclasses
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Literal, Optional
import unittest
from unittest.mock import Mock

try:
    import tyro
except ImportError:
    tyro = None


SERVER_PATH = Path(__file__).resolve().parents[1] / "openvla-oft-cf/serving/serve_openvla_oft.py"


def load_server():
    """Execute the actual Args/main, replacing only external runtime dependencies."""
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = {
        "__name__": __name__,
        "dataclass": dataclasses.dataclass,
        "replace": dataclasses.replace,
        "Literal": Literal,
        "logging": logging,
        "NUM_ACTIONS_CHUNK": 8,
        "OpenVLAOFTPolicy": Mock(),
        "WebsocketPolicyServer": Mock(),
    }
    exec(compile(tree, str(SERVER_PATH), "exec"), namespace)
    return namespace


def load_modality_resolver():
    """Load the real evaluator contract without simulator or test-package imports."""
    path = Path(__file__).with_name("main_cf.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_input_modality"
    ]
    namespace = {
        "logging": logging,
        "Dict": Dict,
        "Optional": Optional,
        "InputModality": Literal["language", "vision", "vision_language"],
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_resolve_input_modality"]


class OpenVLAModalityTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_server()
        self.resolve = load_modality_resolver()
        self.Args = self.namespace["Args"]
        self.loader = self.namespace["OpenVLAOFTPolicy"]
        self.server = self.namespace["WebsocketPolicyServer"]
        # Keep unrelated metadata and conflicting old values to test the
        # handshake's authoritative fields without mutating the shared policy.
        self.original_metadata = {
            "policy_name": "OpenVLA",
            "expects": {"state": {"shape": [8]}},
            "custom_feature": "preserved",
            "input_modality": "old",
            "model_backend": "old",
            "state_input_visible": "old",
        }
        self.metadata_snapshot = copy.deepcopy(self.original_metadata)

        def create_policy(model_path, **kwargs):
            self.policy = SimpleNamespace(
                metadata=self.original_metadata,
                empty_language=kwargs["empty_language"],
                cfg=SimpleNamespace(pretrained_checkpoint=model_path, **kwargs),
            )
            return self.policy

        self.loader.side_effect = create_policy

    def test_modality_is_a_required_dataclass_argument(self):
        field = next(field for field in dataclasses.fields(self.Args) if field.name == "modality")
        self.assertIs(field.default, dataclasses.MISSING)
        self.assertIs(field.default_factory, dataclasses.MISSING)
        with self.assertRaises(TypeError):
            self.Args(model_path="/checkpoint")

    def test_invalid_modality_fails_before_loading_or_starting_server(self):
        for modality in (None, "", "invalid", "Vision"):
            with self.subTest(modality=modality):
                args = self.Args(model_path="/checkpoint", modality=modality)
                snapshot = dataclasses.asdict(args)
                with self.assertRaises(ValueError):
                    self.namespace["main"](args)
                self.assertEqual(dataclasses.asdict(args), snapshot)
        self.loader.assert_not_called()
        self.server.assert_not_called()

    def test_three_modes_and_proprio_preserve_options_and_report_effective_masks(self):
        for modality, empty_language, mask_vision in (
            ("language", False, True),
            ("vision", True, False),
            ("vision_language", False, False),
        ):
            for use_proprio in (False, True):
                with self.subTest(modality=modality, use_proprio=use_proprio):
                    self.loader.reset_mock()
                    self.server.reset_mock()
                    args = self.Args(
                        model_path="/checkpoint",
                        modality=modality,
                        port=8765,
                        device="cuda:2",
                        # Deliberately contradict the requested modality.
                        empty_language=not empty_language,
                        mask_vision_tokens=not mask_vision,
                        task_suite_name="libero_object",
                        num_open_loop_steps=6,
                        use_proprio=use_proprio,
                        num_images_in_input=1,
                        center_crop=False,
                        load_in_8bit=True,
                        load_in_4bit=False,
                        use_l1_regression=False,
                    )
                    snapshot = dataclasses.asdict(args)
                    self.namespace["main"](args)
                    self.assertEqual(dataclasses.asdict(args), snapshot)
                    self.loader.assert_called_once_with(
                        "/checkpoint",
                        device="cuda:2",
                        empty_language=empty_language,
                        task_suite_name="libero_object",
                        num_open_loop_steps=6,
                        use_proprio=use_proprio,
                        num_images_in_input=1,
                        center_crop=False,
                        load_in_8bit=True,
                        load_in_4bit=False,
                        use_l1_regression=False,
                        mask_vision_tokens=mask_vision,
                    )
                    self.server.assert_called_once()
                    options = self.server.call_args.kwargs
                    self.assertIs(options["policy"], self.policy)
                    self.assertEqual(options["host"], "0.0.0.0")
                    self.assertEqual(options["port"], 8765)
                    metadata = options["metadata"]
                    state_visible = use_proprio and modality != "language"
                    self.assertEqual(metadata, {
                        **self.metadata_snapshot,
                        "model_type": "openvla_oft",
                        "model_backend": "pytorch",
                        "input_modality": modality,
                        "empty_language": empty_language,
                        "mask_vision_tokens": mask_vision,
                        "language_ablation_method": "empty_prompt" if modality == "vision" else "none",
                        "vision_ablation_method": "attention_mask" if modality == "language" else "none",
                        "vision_mask_scope": "inserted_patch_span" if modality == "language" else "none",
                        "use_proprio": use_proprio,
                        "state_input_visible": state_visible,
                        "state_input_mode": "continuous" if state_visible else "none",
                        "checkpoint": "/checkpoint",
                        "num_images_in_input": 1,
                        "task_suite_name": "libero_object",
                    })
                    self.assertIs(metadata["state_input_visible"], state_visible)
                    self.assertNotIn("mask_language_tokens", metadata)
                    self.assertIsNot(metadata, self.original_metadata)
                    self.assertEqual(self.original_metadata, self.metadata_snapshot)
                    self.server.return_value.serve_forever.assert_called_once_with()
                    self.assertEqual(self.resolve(metadata, modality), modality)
                    self.assertEqual(self.resolve(metadata, None), modality)

    def test_metadata_reads_actual_policy_settings_without_polluting_shared_metadata(self):
        constructor = self.loader.side_effect

        def adjusted_policy(model_path, **kwargs):
            policy = constructor(model_path, **kwargs)
            # If a policy resolves settings differently, describe its actual
            # settings so the evaluator can detect the modality mismatch.
            policy.empty_language = False
            policy.cfg.mask_vision_tokens = True
            policy.cfg.use_proprio = False
            policy.cfg.num_images_in_input = 3
            policy.cfg.task_suite_name = "libero_10"
            return policy

        self.loader.side_effect = adjusted_policy
        args = self.Args(model_path="/checkpoint", modality="vision", use_proprio=True)
        self.namespace["main"](args)
        metadata = self.server.call_args.kwargs["metadata"]
        self.assertIs(metadata["empty_language"], False)
        self.assertIs(metadata["mask_vision_tokens"], True)
        self.assertEqual(metadata["language_ablation_method"], "none")
        self.assertEqual(metadata["vision_ablation_method"], "attention_mask")
        self.assertEqual(metadata["vision_mask_scope"], "inserted_patch_span")
        self.assertIs(metadata["use_proprio"], False)
        self.assertIs(metadata["state_input_visible"], False)
        self.assertEqual(metadata["state_input_mode"], "none")
        self.assertEqual(metadata["num_images_in_input"], 3)
        self.assertEqual(metadata["task_suite_name"], "libero_10")
        self.assertEqual(self.original_metadata, self.metadata_snapshot)
        with self.assertRaises(ValueError):
            self.resolve(metadata, "vision")
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(self.resolve(metadata, None))

    @unittest.skipIf(tyro is None, "Optional real CLI parsing requires tyro")
    def test_tyro_accepts_explicit_modality(self):
        for modality in ("language", "vision", "vision_language"):
            with self.subTest(modality=modality):
                args = tyro.cli(self.Args, args=["--model-path", "/checkpoint", "--modality", modality])
                self.assertEqual(args.model_path, "/checkpoint")
                self.assertEqual(args.modality, modality)

    @unittest.skipIf(tyro is None, "Optional real CLI parsing requires tyro")
    def test_tyro_rejects_missing_or_invalid_modality(self):
        for options in ([], ["--modality", "invalid"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    tyro.cli(self.Args, args=["--model-path", "/checkpoint", *options])
                self.assertNotEqual(failure.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
