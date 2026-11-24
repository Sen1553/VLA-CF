"""Metadata contract tests that do not import MuJoCo, LIBERO, or model runtimes."""

import ast
import dataclasses
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Dict, Literal, Optional
import unittest
from unittest.mock import Mock

try:
    import numpy as np
except ImportError:
    np = None  # Core metadata checks do not require optional simulator dependencies.


def load_evaluator():
    # Execute only the pure metadata logic and evaluator definitions. Runtime
    # dependencies are supplied by the tests instead of importing a simulator.
    path = Path(__file__).with_name("main_cf.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"Args", "_resolve_input_modality", "_json_metadata_default", "eval_libero"}
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {
        "dataclasses": dataclasses,
        "logging": logging,
        "Dict": Dict,
        "Optional": Optional,
        "InputModality": Literal["language", "vision", "vision_language"],
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def metadata(modality):
    return {
        "input_modality": modality,
        "mask_language_tokens": modality == "vision",
        "mask_vision_tokens": modality == "language",
        "model_backend": "jax",
        "training_config": "pi05_libero",
        "checkpoint": "/test/checkpoint",
        "sampling_num_steps": 10,
    }


def openvla_metadata(modality, use_proprio=True):
    empty_language = modality == "vision"
    mask_vision = modality == "language"
    state_visible = use_proprio and not mask_vision
    return {
        "model_type": "openvla_oft",
        "model_backend": "pytorch",
        "input_modality": modality,
        "empty_language": empty_language,
        "mask_vision_tokens": mask_vision,
        "language_ablation_method": "empty_prompt" if empty_language else "none",
        "vision_ablation_method": "attention_mask" if mask_vision else "none",
        "vision_mask_scope": "inserted_patch_span" if mask_vision else "none",
        "use_proprio": use_proprio,
        "state_input_visible": state_visible,
        "state_input_mode": "continuous" if state_visible else "none",
        "checkpoint": "/test/openvla-checkpoint",
        "num_images_in_input": 2,
    }


class ModalityContractTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_evaluator()
        self.resolve = self.namespace["_resolve_input_modality"]

    def test_all_three_valid_modes(self):
        for mode in ("language", "vision", "vision_language"):
            with self.subTest(mode=mode):
                self.assertEqual(self.resolve(metadata(mode), mode), mode)
                self.assertEqual(self.resolve(metadata(mode), None), mode)

    def test_strict_mode_rejects_missing_or_conflicting_evidence(self):
        for field in ("input_modality", "mask_language_tokens", "mask_vision_tokens", "model_backend"):
            value = metadata("language")
            del value[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.resolve(value, "language")
        for changes in (
            {"input_modality": "vision"},
            {"input_modality": "unknown"},
            {"mask_language_tokens": True},
            {"mask_vision_tokens": False},
            {"mask_vision_tokens": 1},
            {"mask_vision_tokens": "true"},
            {"model_backend": "pytorch"},
        ):
            value = metadata("language")
            value.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve(value, "language")

    def test_legacy_server_stays_unknown(self):
        for value in ({}, None, {"reset_pose": [0, 0]}, {"input_modality": "vision_language"}):
            with self.subTest(value=value), self.assertLogs(level="WARNING"):
                self.assertIsNone(self.resolve(value, None))
        with self.assertRaises(ValueError):
            self.resolve({}, "vision_language")

    def test_unchecked_non_jax_or_missing_backend_stays_unknown(self):
        for backend in ("pytorch", None):
            value = metadata("vision_language")
            if backend is None:
                del value["model_backend"]
            else:
                value["model_backend"] = backend
            with self.subTest(backend=backend), self.assertLogs(level="WARNING"):
                self.assertIsNone(self.resolve(value, None))

    def test_pi0_and_pi05_do_not_gain_pytorch_mask_support(self):
        for model_type in ("pi0", "pi05", "other"):
            for mode in ("language", "vision", "vision_language"):
                value = metadata(mode)
                value.update({"model_type": model_type, "model_backend": "pytorch"})
                with self.subTest(model_type=model_type, modality=mode):
                    with self.assertRaises(ValueError):
                        self.resolve(value, mode)
                    with self.assertLogs(level="WARNING"):
                        self.assertIsNone(self.resolve(value, None))

    def test_openvla_rejects_missing_contract_fields(self):
        for field in (
            "model_type", "model_backend", "input_modality", "empty_language", "mask_vision_tokens",
            "language_ablation_method", "vision_ablation_method", "vision_mask_scope",
            "use_proprio", "state_input_visible", "state_input_mode",
        ):
            value = openvla_metadata("vision")
            del value[field]
            with self.subTest(missing=field):
                with self.assertRaises(ValueError):
                    self.resolve(value, "vision")
                with self.assertLogs(level="WARNING"):
                    self.assertIsNone(self.resolve(value, None))

    def test_openvla_rejects_conflicting_flags_methods_and_state(self):
        for mode in ("language", "vision", "vision_language"):
            valid = openvla_metadata(mode)
            mutations = [
                {"model_backend": "jax"},
                {"input_modality": "unknown"},
                {"empty_language": not valid["empty_language"]},
                {"empty_language": int(valid["empty_language"])},
                {"empty_language": "true"},
                {"mask_vision_tokens": not valid["mask_vision_tokens"]},
                {"mask_vision_tokens": int(valid["mask_vision_tokens"])},
                {"language_ablation_method": "attention_mask"},
                {"vision_ablation_method": "zero_image"},
                {"vision_mask_scope": "vision_only"},
                {"use_proprio": 1},
                {"state_input_visible": not valid["state_input_visible"]},
                {"state_input_visible": int(valid["state_input_visible"])},
                {"state_input_mode": "none" if valid["state_input_visible"] else "continuous"},
            ]
            for change in mutations:
                value = {**valid, **change}
                with self.subTest(modality=mode, change=change):
                    with self.assertRaises(ValueError):
                        self.resolve(value, mode)
                    with self.assertLogs(level="WARNING"):
                        self.assertIsNone(self.resolve(value, None))
            mismatched_expectation = "language" if mode != "language" else "vision"
            with self.subTest(modality=mode, expected=mismatched_expectation), self.assertRaises(ValueError):
                self.resolve(valid, mismatched_expectation)

    def run_empty_suite(self, server_metadata, expected_modality):
        namespace = self.namespace
        client = SimpleNamespace(get_server_metadata=lambda: server_metadata)
        suite = SimpleNamespace(name="libero_cf_probe", n_tasks=0, get_task_max_steps=lambda _: 220)
        namespace.update({
            "np": np if np is not None else SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
            "benchmark": SimpleNamespace(get_benchmark_dict=lambda: {"libero-cf-probe": lambda: suite}),
            "_websocket_client_policy": SimpleNamespace(WebsocketClientPolicy=lambda *_: client),
            "datetime": SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: "timestamp")),
            "pathlib": __import__("pathlib"),
            "json": json,
            "tqdm": SimpleNamespace(tqdm=lambda value: value),
            "_setup_logging": Mock(),
            "_load_custom_conditions_db": lambda _: {},
        })
        args = namespace["Args"](task_suite_name="libero-cf-probe", expected_modality=expected_modality)
        namespace["eval_libero"](args)

    def test_fail_before_creating_rollout_outputs(self):
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                with self.assertRaises(ValueError):
                    self.run_empty_suite(metadata("vision"), "language")
                self.assertFalse(Path("rollouts").exists())
            finally:
                os.chdir(previous_dir)

    def test_manifest_and_directory_use_verified_modality(self):
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                expected = metadata("language")
                self.run_empty_suite(expected, "language")
                path = Path("rollouts/libero_cf_probe-language-timestamp/run_config.json")
                recorded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(recorded["server_metadata"], expected)
                self.assertEqual(recorded["input_modality"], "language")
                self.assertEqual(recorded["args"]["expected_modality"], "language")
                self.assertEqual(recorded["args"]["task_suite_name"], "libero_cf_probe")
                with self.assertLogs(level="WARNING"):
                    self.run_empty_suite({}, None)
                path = Path("rollouts/libero_cf_probe-timestamp/run_config.json")
                self.assertIsNone(json.loads(path.read_text(encoding="utf-8"))["input_modality"])
            finally:
                os.chdir(previous_dir)

    def test_pi0_and_pi05_metadata_are_accepted_and_saved_for_each_modality(self):
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                for model_type, state_mode in (("pi0", "continuous"), ("pi05", "none")):
                    for mode in ("language", "vision", "vision_language"):
                        with self.subTest(model_type=model_type, modality=mode):
                            expected = metadata(mode)
                            expected.update({
                                "training_config": model_type + "_libero",
                                "model_type": model_type,
                                "state_input_mode": state_mode,
                            })
                            self.assertEqual(self.resolve(expected, mode), mode)
                            self.run_empty_suite(expected, mode)
                            path = Path("rollouts/libero_cf_probe-{}-timestamp/run_config.json".format(mode))
                            recorded = json.loads(path.read_text(encoding="utf-8"))
                            self.assertEqual(recorded["server_metadata"], expected)
                            self.assertEqual(recorded["input_modality"], mode)
            finally:
                os.chdir(previous_dir)

    def test_openvla_modalities_and_proprio_states_are_verified_and_saved(self):
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                for mode in ("language", "vision", "vision_language"):
                    for use_proprio in (False, True):
                        with self.subTest(modality=mode, use_proprio=use_proprio):
                            expected = openvla_metadata(mode, use_proprio)
                            self.assertNotIn("mask_language_tokens", expected)
                            self.assertEqual(self.resolve(expected, mode), mode)
                            self.assertEqual(self.resolve(expected, None), mode)
                            self.run_empty_suite(expected, mode)
                            path = Path("rollouts/libero_cf_probe-{}-timestamp/run_config.json".format(mode))
                            recorded = json.loads(path.read_text(encoding="utf-8"))
                            self.assertEqual(recorded["server_metadata"], expected)
                            self.assertEqual(recorded["input_modality"], mode)
                            self.assertEqual(recorded["args"]["expected_modality"], mode)
            finally:
                os.chdir(previous_dir)

    def test_legacy_openvla_is_unknown_and_strict_failure_precedes_outputs(self):
        legacy = {"policy_name": "OpenVLA", "expects": {"prompt": {"type": "str"}}}
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                with self.assertRaises(ValueError):
                    self.run_empty_suite(legacy, "vision_language")
                self.assertFalse(Path("rollouts").exists())
                with self.assertLogs(level="WARNING"):
                    self.run_empty_suite(legacy, None)
                path = Path("rollouts/libero_cf_probe-timestamp/run_config.json")
                recorded = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsNone(recorded["input_modality"])
                self.assertEqual(recorded["server_metadata"], legacy)
            finally:
                os.chdir(previous_dir)

    @unittest.skipIf(np is None, "NumPy is required only for NumPy metadata serialization")
    def test_legacy_numpy_metadata_is_preserved_in_manifest(self):
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                value = {
                    "reset_pose": np.array([[0.0, 1.25]], dtype=np.float32),
                    "count": np.int64(3),
                    "scale": np.float32(1.25),
                    "enabled": np.bool_(True),
                }
                with self.assertLogs(level="WARNING"):
                    self.run_empty_suite(value, None)
                path = Path("rollouts/libero_cf_probe-timestamp/run_config.json")
                recorded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(recorded["server_metadata"], {
                    "reset_pose": [[0.0, 1.25]], "count": 3, "scale": 1.25, "enabled": True,
                })
                self.assertIsNone(recorded["input_modality"])
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
