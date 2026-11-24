"""Service modality contracts, isolated from JAX, Torch, Flax, and checkpoints.

The test file can be collected on Python 3.8. Executing the actual service AST
requires Python 3.11, matching the service runtime rather than the evaluator.
"""

import ast
import dataclasses
import enum
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "openpi-cf"


class ModelType(enum.Enum):
    PI0 = "pi0"
    PI05 = "pi05"


@dataclasses.dataclass(frozen=True)
class TrainConfigStub:
    model: object
    policy_metadata: dict = dataclasses.field(default_factory=lambda: {"custom": "preserved"})
    name: str = "original_config"


def execute_definitions(path, namespace, names=None):
    """Discard imports and the CLI guard while preserving the real functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (names is None or node.name in names):
            definitions.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "DEFAULT_CHECKPOINT":
                definitions.append(node)
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + definitions
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def load_real_pi0_config():
    """Keep actual config defaults/validation, excluding heavyweight methods."""
    path = OPENPI / "src/openpi/models/pi0_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pi0Config")
    config_class.body = [
        node for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        or (isinstance(node, ast.FunctionDef) and node.name in ("__post_init__", "model_type"))
    ]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), config_class]
    ast.fix_missing_locations(tree)
    namespace = {
        "__name__": __name__, "dataclasses": dataclasses,
        "_model": SimpleNamespace(BaseModelConfig=object, ModelType=ModelType), "override": lambda function: function,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Pi0Config"]


def load_server():
    namespace = {
        "__name__": __name__, "dataclasses": dataclasses, "enum": enum, "logging": logging, "Literal": Literal,
        "_pi0_config": SimpleNamespace(Pi0Config=load_real_pi0_config()),
        "_config": SimpleNamespace(get_config=Mock()),
        "_policy_config": SimpleNamespace(create_trained_policy=Mock()),
        "_policy": SimpleNamespace(Policy=object, PolicyRecorder=Mock()),
        "socket": SimpleNamespace(gethostname=lambda: "test-host", gethostbyname=lambda _: "127.0.0.1"),
        "websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=Mock()),
    }
    return execute_definitions(OPENPI / "scripts/serve_policy.py", namespace)


@unittest.skipIf(sys.version_info < (3, 11), "The service source requires Python 3.11; evaluator tests remain Python 3.8 compatible")
class ServePolicyModalityTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_server()
        self.Pi0Config = self.namespace["_pi0_config"].Pi0Config
        self.lookup = self.namespace["_config"].get_config
        self.loader = self.namespace["_policy_config"].create_trained_policy
        self.loader.return_value = SimpleNamespace(metadata={"model_backend": "jax"})

    def checkpoint_args(self, config_name="pi0_libero", modality=None, **kwargs):
        return self.namespace["Args"](
            policy=self.namespace["Checkpoint"](config=config_name, dir="/checkpoint"),
            modality=modality, **kwargs
        )

    def test_both_models_and_three_modalities_reach_loader_without_mutating_config(self):
        for pi05, model_type, state_mode in ((False, "pi0", "continuous"), (True, "pi05", "none")):
            for modality, expected_language_mask, expected_vision_mask in (
                ("language", False, True), ("vision", True, False), ("vision_language", False, False),
            ):
                with self.subTest(model_type=model_type, modality=modality):
                    model = self.Pi0Config(
                        pi05=pi05, discrete_state_input=False,
                        mask_language_tokens=True, mask_vision_tokens=True,
                        empty_language_probability=0.25,
                    )
                    original = TrainConfigStub(model=model)
                    snapshot = dataclasses.asdict(original)
                    self.lookup.return_value = original
                    self.loader.reset_mock()
                    args = self.checkpoint_args(
                        model_type + "_libero", modality,
                        default_prompt="test prompt", overwrite=True, probability=0.4,
                    )
                    result = self.namespace["create_policy"](args)
                    self.assertIs(result, self.loader.return_value)
                    self.loader.assert_called_once()
                    loaded, checkpoint = self.loader.call_args.args
                    options = self.loader.call_args.kwargs
                    self.assertEqual(checkpoint, "/checkpoint")
                    self.assertIsNot(loaded, original)
                    self.assertIsNot(loaded.model, model)
                    self.assertEqual(dataclasses.asdict(original), snapshot)
                    self.assertEqual(loaded.model.mask_language_tokens, expected_language_mask)
                    self.assertEqual(loaded.model.mask_vision_tokens, expected_vision_mask)
                    self.assertEqual(loaded.model.empty_language_probability, 0.25)
                    self.assertEqual(loaded.model.pi05, pi05)
                    self.assertEqual(loaded.policy_metadata, {
                        "custom": "preserved", "training_config": model_type + "_libero", "checkpoint": "/checkpoint",
                        "input_modality": modality, "mask_language_tokens": expected_language_mask,
                        "mask_vision_tokens": expected_vision_mask, "sampling_num_steps": 10,
                        "model_type": model_type, "state_input_mode": state_mode,
                    })
                    self.assertEqual(options, {
                        "default_prompt": "test prompt", "overwrite": True, "probability": 0.4,
                        "sample_kwargs": {"num_steps": 10}, "require_jax": True,
                    })

    def test_no_modality_preserves_existing_model_and_sampling_behavior(self):
        for model in (
            self.Pi0Config(pi05=False, mask_language_tokens=True),
            self.Pi0Config(pi05=True),  # The ordinary pi05 discrete-state config stays allowed.
            SimpleNamespace(unrelated_model=True),
        ):
            with self.subTest(model=model):
                original = TrainConfigStub(model=model)
                self.lookup.return_value = original
                self.namespace["create_policy"](self.checkpoint_args())
                loaded = self.loader.call_args.args[0]
                self.assertIs(loaded.model, original.model)
                self.assertEqual(original.policy_metadata, {"custom": "preserved"})
                self.assertNotIn("input_modality", loaded.policy_metadata)
                self.assertIsNone(self.loader.call_args.kwargs["sample_kwargs"])
                self.assertFalse(self.loader.call_args.kwargs["require_jax"])

    def test_invalid_requests_are_rejected_before_loading(self):
        cases = [
            (self.Pi0Config(), "invalid"),
            (SimpleNamespace(pi05=True, discrete_state_input=False), "language"),
        ]
        cases.extend((self.Pi0Config(pi05=True), mode) for mode in ("language", "vision", "vision_language"))
        for model, modality in cases:
            with self.subTest(model=model, modality=modality):
                self.lookup.return_value = TrainConfigStub(model=model)
                self.loader.reset_mock()
                with self.assertRaises(ValueError):
                    self.namespace["create_policy"](self.checkpoint_args(modality=modality))
                self.loader.assert_not_called()

    def test_discrete_state_rejection_is_specific_to_pi05(self):
        self.lookup.return_value = TrainConfigStub(model=self.Pi0Config(pi05=False, discrete_state_input=True))
        self.namespace["create_policy"](self.checkpoint_args(modality="vision"))
        self.loader.assert_called_once()
        self.assertEqual(self.loader.call_args.args[0].policy_metadata["state_input_mode"], "continuous")

    def test_default_libero_stays_pi05_and_handshake_exposes_policy_metadata(self):
        self.lookup.return_value = TrainConfigStub(model=self.Pi0Config(pi05=True, discrete_state_input=False))

        def policy_from_config(config, checkpoint, **kwargs):
            return SimpleNamespace(metadata={**config.policy_metadata, "model_backend": "jax"})

        self.loader.side_effect = policy_from_config
        args = self.namespace["Args"](env=self.namespace["EnvMode"].LIBERO, modality="vision", port=8765)
        self.namespace["main"](args)
        self.lookup.assert_called_once_with("pi05_libero")
        self.assertEqual(self.loader.call_args.args[1], "gs://openpi-assets/checkpoints/pi05_libero")
        server_constructor = self.namespace["websocket_policy_server"].WebsocketPolicyServer
        server_constructor.assert_called_once()
        options = server_constructor.call_args.kwargs
        self.assertEqual(options["metadata"]["model_type"], "pi05")
        self.assertEqual(options["metadata"]["input_modality"], "vision")
        self.assertEqual(options["metadata"]["state_input_mode"], "none")
        self.assertEqual(options["metadata"]["model_backend"], "jax")
        self.assertEqual(options["port"], 8765)
        self.assertIs(options["metadata"], options["policy"].metadata)
        server_constructor.return_value.serve_forever.assert_called_once_with()

    def test_jax_requirement_rejects_pytorch_before_any_model_load(self):
        restore = Mock()
        fake_model = SimpleNamespace(
            mask_language_tokens=False, mask_vision_tokens=False, load=Mock(), load_pytorch=Mock(),
        )
        path_exists = Mock(return_value=True)
        namespace = {
            "logging": logging, "pathlib": __import__("pathlib"), "Any": Any,
            "os": SimpleNamespace(path=SimpleNamespace(join=lambda *parts: "/".join(map(str, parts)), exists=path_exists)),
            "download": SimpleNamespace(maybe_download=lambda _: Path("/checkpoint")),
            "transforms": SimpleNamespace(Group=lambda: SimpleNamespace(inputs=[], outputs=[])),
            "_model": SimpleNamespace(restore_params=restore),
            "jnp": SimpleNamespace(bfloat16="bfloat16"),
        }
        execute_definitions(
            OPENPI / "src/openpi/policies/policy_config.py", namespace, {"create_trained_policy"},
        )
        with self.assertRaisesRegex(ValueError, "JAX checkpoint"):
            namespace["create_trained_policy"](SimpleNamespace(model=fake_model), "/checkpoint", require_jax=True)
        path_exists.assert_called_once()
        self.assertTrue(str(path_exists.call_args.args[0]).endswith("model.safetensors"))
        restore.assert_not_called()
        fake_model.load.assert_not_called()
        fake_model.load_pytorch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
