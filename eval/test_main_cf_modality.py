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
