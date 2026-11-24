"""Task-outcome contract tests without simulator or model-runtime imports."""

import ast
import collections
import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import pathlib
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, Literal, Optional
import unittest
from unittest.mock import Mock

try:
    import numpy as np
except ImportError:
    np = None  # Only the evaluator stub tests need NumPy.


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS_DIR = ROOT / "libero" / "libero" / "conditions"
BDDL_DIR = ROOT / "libero" / "libero" / "bddl_files"
SUITES = {"libero_cf_spatial": 15, "libero_cf_object": 10, "libero_cf_ood": 15}


def load_metrics_module():
    """Import the pure helper directly, without importing LIBERO packages."""
    path = Path(__file__).with_name("task_outcome_metrics.py")
    module_spec = importlib.util.spec_from_file_location("tested_task_outcome_metrics", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bddl_sections(path):
    """Read S-expressions only; do not import BDDL or initialize a simulator."""
    source = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8"))
    stack = []
    root = None
    for token in re.findall(r"\(|\)|[^\s()]+", source):
        if token == "(":
            value = []
            if stack:
                stack[-1].append(value)
            else:
                root = value
            stack.append(value)
        elif token == ")":
            stack.pop()
        else:
            stack[-1].append(token)
    if stack or not root or root[0] != "define":
        raise ValueError("Malformed BDDL: {}".format(path))
    return {part[0]: part[1:] for part in root[1:] if isinstance(part, list)}


def declared_instances(tokens):
    names = set()
    index = 0
    while index < len(tokens):
        if tokens[index] == "-":
            index += 2  # Skip the type name, not an object instance.
        else:
            names.add(tokens[index])
            index += 1
    return names


def make_spec():
    return {
        "following_condition": "on apple_1 plate_1",
        "original_training": {
            "suite": "libero_spatial",
            "task": "source_task",
            "condition": "on bowl_1 plate_1",
        },
        "conditions": ["on apple_1 plate_1", "on bowl_1 plate_1", "on cup_1 plate_1"],
    }


class RepositoryTaskMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics_module()

    def load_suite(self, suite):
        db = read_json(CONDITIONS_DIR / (suite + ".json"))
        return self.metrics.load_task_metadata(CONDITIONS_DIR, suite, db), db

    def test_all_40_tasks_have_roles_and_real_scene_entities(self):
        total_tasks = total_conditions = 0
        for suite, expected_count in SUITES.items():
            tasks, db = self.load_suite(suite)
            expected_keys = {
                "{}/{}".format(suite, path.name)
                for path in (BDDL_DIR / suite).glob("*.bddl")
            }
            self.assertEqual(set(tasks), expected_keys)
            self.assertEqual(set(tasks), set(db))
            self.assertEqual(len(tasks), expected_count)
            total_tasks += len(tasks)
            for task_key, spec in tasks.items():
                with self.subTest(task=task_key):
                    conditions = spec["conditions"]
                    self.assertEqual(set(conditions), set(db[task_key]))
                    self.assertEqual(len(conditions), len(set(conditions)))
                    self.assertIn(spec["following_condition"], conditions)
                    original = spec["original_training"]
                    self.assertIn(original["condition"], conditions)
                    self.assertNotEqual(spec["following_condition"], original["condition"])

                    sections = bddl_sections(BDDL_DIR / task_key)
                    objects = declared_instances(sections[":objects"])
                    fixtures = declared_instances(sections.get(":fixtures", []))
                    regions = {}
                    for region in sections.get(":regions", []):
                        region_fields = {item[0]: item[1:] for item in region[1:]}
                        target = region_fields[":target"][0]
                        regions[target + "_" + region[0]] = target
                    entities = objects | fixtures | set(regions)
                    subjects = set()
                    destinations = set()
                    for condition in conditions:
                        predicate, subject, destination = condition.split()
                        self.assertIn(predicate, ("on", "in"))
                        self.assertIn(subject, objects)
                        self.assertIn(destination, entities)
                        subjects.add(subject)
                        destinations.add(destination)
                    self.assertEqual(len(destinations), 1)
                    destination = next(iter(destinations))
                    container = regions.get(destination, destination)
                    self.assertNotIn(container, subjects)
                    self.assertEqual(subjects, objects - {container})
                    self.assertEqual(len(conditions), 6 if suite == "libero_cf_object" else 4)

                    original_path = BDDL_DIR / original["suite"] / (original["task"] + ".bddl")
                    original_goal = bddl_sections(original_path)[":goal"][0]
                    self.assertEqual(original_goal[0].lower(), "and")
                    self.assertEqual(
                        [" ".join(atom).lower() for atom in original_goal[1:]],
                        [original["condition"]],
                    )
                    total_conditions += len(conditions)
        self.assertEqual(total_tasks, 40)
        self.assertEqual(total_conditions, 180)

    def test_spatial_13_uses_bowl_2_for_following_and_bowl_1_for_original(self):
        tasks, _ = self.load_suite("libero_cf_spatial")
        key = next(key for key in tasks if "/13-" in key)
        self.assertEqual(tasks[key]["following_condition"], "on akita_black_bowl_2 plate_1")
        self.assertEqual(tasks[key]["original_training"]["condition"], "on akita_black_bowl_1 plate_1")
        self.assertEqual(
            tasks[key]["original_training"]["task"],
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        )

    def test_ood_source_mapping_is_explicit_and_not_inferred_from_cf_number(self):
        tasks, _ = self.load_suite("libero_cf_ood")
        relations = [
            "between_the_plate_and_the_ramekin", "between_the_plate_and_the_ramekin",
            "next_to_the_ramekin", "from_table_center", "on_the_cookie_box",
            "in_the_top_drawer_of_the_wooden_cabinet", "on_the_ramekin",
            "next_to_the_cookie_box", "on_the_stove", "next_to_the_plate",
            "on_the_wooden_cabinet", "from_table_center", "on_the_cookie_box",
            "in_the_top_drawer_of_the_wooden_cabinet", "next_to_the_plate",
        ]
        following_subjects = ["coke_1"] * 3 + ["cup_1"] * 3 + ["beer_1"] * 3 + ["apple_1"] * 3 + ["banana_1"] * 3
        for index, (relation, subject) in enumerate(zip(relations, following_subjects), 1):
            task_key = next(key for key in tasks if "/{:02d}-".format(index) in key)
            spec = tasks[task_key]
            with self.subTest(task=task_key):
                self.assertEqual(spec["following_condition"], "on {} plate_1".format(subject))
                self.assertEqual(spec["original_training"]["suite"], "libero_spatial")
                self.assertEqual(spec["original_training"]["condition"], "on akita_black_bowl_1 plate_1")
                self.assertEqual(
                    spec["original_training"]["task"],
                    "pick_up_the_black_bowl_{}_and_place_it_on_the_plate".format(relation),
                )

    def test_metadata_mismatches_fail_before_scoring(self):
        suite = "libero_cf_spatial"
        db = read_json(CONDITIONS_DIR / (suite + ".json"))
        source = read_json(CONDITIONS_DIR / "metadata" / (suite + ".json"))
        key = next(iter(source["tasks"]))
        changed = []
        value = copy.deepcopy(source)
        value["suite"] = "libero_cf_object"
        changed.append(value)
        value = copy.deepcopy(source)
        del value["tasks"][key]
        changed.append(value)
        value = copy.deepcopy(source)
        value["tasks"][key]["following_condition"] = "on absent_1 plate_1"
        changed.append(value)
        value = copy.deepcopy(source)
        value["tasks"][key]["original_training"]["condition"] = "on absent_1 plate_1"
        changed.append(value)
        value = copy.deepcopy(source)
        value["tasks"][key]["conditions"].pop()
        changed.append(value)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary_dir:
            directory = Path(temporary_dir)
            (directory / "metadata").mkdir()
            path = directory / "metadata" / (suite + ".json")
            for index, value in enumerate(changed):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(case=index), self.assertRaises(ValueError):
                    self.metrics.load_task_metadata(directory, suite, db)


class TaskOutcomeMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics_module()

    def setUp(self):
        self.spec = make_spec()
        self.following, self.original, self.other = self.spec["conditions"]

    def episode(self, successes=(), touches=(), **kwargs):
        results = {condition: condition in successes for condition in self.spec["conditions"]}
        touched = {condition: condition in touches for condition in self.spec["conditions"]}
        kwargs.setdefault("end_reason", "candidate_success" if successes else "time_limit")
        return self.metrics.build_episode_result(self.spec, results, touched, **kwargs)

    def test_all_four_valid_outcomes_and_original_is_a_deviation_subset(self):
        cases = [
            ((), "failure", False, False),
            ((self.following,), "following_only", True, False),
            ((self.original,), "deviation_only", False, True),
            ((self.other,), "deviation_only", False, True),
            ((self.following, self.other), "both", True, True),
        ]
        for successes, outcome, follows, deviates in cases:
            with self.subTest(outcome=outcome, successes=successes):
                result = self.episode(successes)
                self.assertEqual(result["status"], "valid")
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["following"]["success"], follows)
                self.assertEqual(result["deviation"]["success"], deviates)
                self.assertEqual(set(result["deviation"]["by_goal"]), {self.original, self.other})
                self.assertEqual(result["deviation"]["original_training"]["success"], self.original in successes)
                self.assertEqual(len(result["per_condition"]), 3)

    def test_multiple_deviations_count_once_but_touch_rates_sum_to_two(self):
        accumulator = self.metrics.TaskMetrics("test/task.bddl", self.spec)
        result = self.episode((self.original, self.other), (self.original, self.other))
        accumulator.add_episode(result)
        summary = accumulator.summary()
        self.assertEqual(summary["counts"]["valid"], 1)
        self.assertEqual(summary["outcomes"]["deviation_only"], 1)
        self.assertEqual(summary["deviation"]["successes"], 1)
        self.assertEqual(summary["deviation"]["success_rate"], 1.0)
        self.assertEqual(summary["deviation"]["original_training"]["successes"], 1)
        self.assertEqual(summary["deviation"]["by_goal"][self.original]["successes"], 1)
        self.assertEqual(summary["deviation"]["by_goal"][self.other]["successes"], 1)
        self.assertEqual(summary["deviation"]["touch_rate_sum"], 2.0)
        self.assertEqual(set(result["deviation"]["successful_conditions"]), {self.original, self.other})

    def test_both_is_inclusive_deviation_but_not_deviation_only(self):
        accumulator = self.metrics.TaskMetrics("test/task.bddl", self.spec)
        accumulator.add_episode(self.episode((self.following, self.original)))
        summary = accumulator.summary()
        self.assertEqual(summary["following"]["success_rate"], 1.0)
        self.assertEqual(summary["deviation"]["success_rate"], 1.0)
        self.assertEqual(summary["deviation"]["only_success_rate"], 0.0)
        self.assertEqual(summary["outcomes"]["both"], 1)

    def test_error_and_invalid_initial_are_excluded_from_valid_denominators(self):
        accumulator = self.metrics.TaskMetrics("test/task.bddl", self.spec)
        accumulator.add_episode(self.episode((self.following,), (self.following,)))
        for reason in ("error", "invalid_initial"):
            result = self.metrics.build_episode_result(
                self.spec, {}, {}, end_reason=reason,
                error="simulator unavailable" if reason == "error" else None,
            )
            self.assertEqual(result["status"], reason)
            self.assertEqual(result["outcome"], reason)
            self.assertFalse(result["touch_valid"])
            self.assertIsNone(result["following"]["success"])
            self.assertIsNone(result["following"]["touched"])
            self.assertIsNone(result["deviation"]["success"])
            self.assertIsNone(result["deviation"]["successful_conditions"])
            for goal in result["per_condition"]:
                self.assertIsNone(goal["success"])
                self.assertIsNone(goal["touched"])
            accumulator.add_episode(result)
        summary = accumulator.summary()
        self.assertEqual(summary["counts"], {"attempted": 3, "valid": 1, "error": 1, "invalid_initial": 1, "touch_valid": 1})
        self.assertEqual(summary["following"]["success_rate"], 1.0)
        self.assertEqual(summary["following"]["touch_rate"], 1.0)
        self.assertEqual(summary["outcomes"]["failure"], 0)

    def test_unknown_touch_is_null_and_uses_its_own_denominator(self):
        accumulator = self.metrics.TaskMetrics("test/task.bddl", self.spec)
        values = {condition: condition == self.following for condition in self.spec["conditions"]}
        unknown = self.metrics.build_episode_result(
            self.spec, values, {}, end_reason="candidate_success", touch_valid=False,
        )
        accumulator.add_episode(unknown)
        self.assertIsNone(unknown["following"]["touched"])
        self.assertIsNone(accumulator.summary()["following"]["touch_rate"])
        self.assertIsNone(accumulator.summary()["deviation"]["touch_rate_sum"])
        accumulator.add_episode(self.episode((), (self.following, self.original, self.other)))
        summary = accumulator.summary()
        self.assertEqual(summary["counts"]["valid"], 2)
        self.assertEqual(summary["counts"]["touch_valid"], 1)
        self.assertEqual(summary["following"]["success_rate"], 0.5)
        self.assertEqual(summary["following"]["touch_rate"], 1.0)
        self.assertEqual(summary["deviation"]["touch_rate_sum"], 2.0)

    def test_empty_denominators_are_null(self):
        summary = self.metrics.TaskMetrics("test/task.bddl", self.spec).summary()
        self.assertEqual(summary["counts"]["attempted"], 0)
        self.assertIsNone(summary["following"]["success_rate"])
        self.assertIsNone(summary["following"]["touch_rate"])
        self.assertIsNone(summary["deviation"]["success_rate"])
        self.assertIsNone(summary["deviation"]["touch_rate_sum"])

    def test_incomplete_or_non_boolean_measurements_are_rejected(self):
        valid = {condition: False for condition in self.spec["conditions"]}
        invalid = [{}, {**valid, self.original: None}, {**valid, self.original: 1}]
        for values in invalid:
            with self.subTest(results=values), self.assertRaises(ValueError):
                self.metrics.build_episode_result(self.spec, values, valid, end_reason="time_limit")
            with self.subTest(touched=values), self.assertRaises(ValueError):
                self.metrics.build_episode_result(self.spec, valid, values, end_reason="time_limit")

    def test_macro_and_pooled_use_different_weights_and_exclude_invalid_tasks(self):
        task_a, task_b, task_c = ["libero_cf_spatial/{}.bddl".format(name) for name in ("a", "b", "c")]
        first = self.metrics.TaskMetrics(task_a, self.spec)
        first.add_episode(self.episode((self.following,), (self.original, self.other)))
        second = self.metrics.TaskMetrics(task_b, self.spec)
        for _ in range(3):
            second.add_episode(self.episode())
        invalid = self.metrics.TaskMetrics(task_c, self.spec)
        invalid.add_episode(self.metrics.build_episode_result(self.spec, {}, {}, end_reason="error"))
        summary = self.metrics.summarize_run("libero_cf_spatial", {
            task_a: first.summary(), task_b: second.summary(), task_c: invalid.summary(),
        })
        self.assertEqual(set(summary["tasks"]), {task_a, task_b, task_c})
        self.assertEqual(summary["pooled"]["counts"]["attempted"], 5)
        self.assertEqual(summary["pooled"]["counts"]["valid"], 4)
        self.assertEqual(summary["pooled"]["following"]["success_rate"], 0.25)
        self.assertEqual(summary["macro"]["following"]["success_rate"], 0.5)
        self.assertEqual(summary["pooled"]["deviation"]["touch_rate_sum"], 0.5)
        self.assertEqual(summary["macro"]["deviation"]["touch_rate_sum"], 1.0)
        self.assertEqual(summary["macro"]["valid_task_counts"], {"success": 2, "touch": 2})


def load_evaluator_definitions():
    """Run the real evaluator function with only its external services stubbed."""
    path = Path(__file__).with_name("main_cf.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        "Args", "_resolve_input_modality", "_json_metadata_default", "eval_libero",
        "_write_outcome_summary", "_format_metric_rate", "_log_task_outcomes",
        "_condition_subject_token", "_get_conditions_for_task",
        "_get_gripper_contact_body_names", "_resolve_instance_root_body_name",
    }
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {
        "dataclasses": dataclasses, "logging": Mock(), "Dict": Dict, "Optional": Optional,
        "InputModality": Literal["language", "vision", "vision_language"],
        "pathlib": pathlib, "json": json, "os": os, "collections": collections, "np": np,
        "_task_metrics": load_metrics_module(),
        "LIBERO_DUMMY_ACTION": [0.0] * 6 + [-1.0], "LIBERO_ENV_RESOLUTION": 256,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def contact_sim(subjects, touched=()):
    """A gripper and object child bodies, with both contact argument orders."""
    names = ["world", "robot0", "robot0_gripper_left_finger"]
    parents = [0, 0, 1]
    geom_bodies = [2]
    for subject in subjects:
        root_id = len(names)
        names.extend([subject, subject + "_collision_child"])
        parents.extend([0, root_id])
        geom_bodies.append(root_id + 1)
    contacts = []
    for index, subject in enumerate(subjects):
        if subject in touched:
            pair = (0, index + 1) if index % 2 == 0 else (index + 1, 0)
            contacts.append(SimpleNamespace(geom1=pair[0], geom2=pair[1]))
    model = SimpleNamespace(
        body_id2name=lambda index: names[index], body_name2id=lambda name: names.index(name),
        body_names=names, body_parentid=parents, geom_bodyid=geom_bodies, nbody=len(names),
    )
    return SimpleNamespace(model=model, data=SimpleNamespace(ncon=len(contacts), contact=contacts))


class ContactObservationTest(unittest.TestCase):
    def test_strict_keeps_existing_root_body_contact_semantics(self):
        helper = load_evaluator_definitions()["_get_gripper_contact_body_names"]
        sim = contact_sim(["apple_1", "bowl_1", "cup_1"], ["apple_1", "bowl_1"])
        # Object-to-object contact must not count as a gripper touch.
        sim.data.contact.append(SimpleNamespace(geom1=2, geom2=3))
        sim.data.ncon += 1
        env = SimpleNamespace(sim=sim)
        self.assertEqual(helper(env), {"apple_1", "bowl_1"})
        self.assertEqual(helper(env, strict=True), {"apple_1", "bowl_1"})
        sim.data = None
        self.assertEqual(helper(env), set())
        with self.assertRaisesRegex(ValueError, "contact data unavailable"):
            helper(env, strict=True)


@unittest.skipIf(np is None, "NumPy is required for the evaluator observation/action stub")
class EvaluatorOutcomeIntegrationTest(unittest.TestCase):
    def run_scenarios(self, scenarios, stop_on_error=False, media_error=False):
        namespace = load_evaluator_definitions()
        suite_name = "libero_cf_object"
        db = read_json(CONDITIONS_DIR / (suite_name + ".json"))
        task_key = next(key for key in db if "/05-" in key)
        spec = namespace["_task_metrics"].load_task_metadata(CONDITIONS_DIR, suite_name, db)[task_key]
        following = spec["following_condition"]
        original = spec["original_training"]["condition"]
        other = next(condition for condition in spec["conditions"] if condition not in (following, original))
        roles = {"following": following, "original": original, "other": other}
        subjects = [condition.split()[1] for condition in spec["conditions"]]
        plans = []
        for scenario in scenarios:
            plan = dict(scenario)
            for field in ("successes", "initial", "touches"):
                plan[field] = [roles[role] for role in plan.get(field, [])]
            plans.append(plan)
        wait_steps = 1

        class Environment:
            def __init__(self):
                self.episode_index = -1
                self.sim = contact_sim(subjects)
                self.steps = collections.Counter()
                self.infers = collections.Counter()
                self.closed = False

            @property
            def plan(self):
                return plans[self.episode_index]

            def reset(self):
                self.episode_index += 1
                self.sim = contact_sim(subjects)
                if self.plan.get("reset_error"):
                    raise RuntimeError("reset failed")

            def observation(self):
                return {
                    "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                    "robot0_gripper_qpos": np.zeros(2),
                }

            def set_init_state(self, state):
                if state != self.episode_index:
                    raise AssertionError("Wrong episode initialization")
                return self.observation()

            def set_custom_conditions(self, conditions):
                pass

            def set_success_any_conditions(self, conditions):
                self.conditions = conditions

            def evaluate_conditions(self, conditions):
                field = "initial" if self.steps[self.episode_index] <= wait_steps else "successes"
                return {condition: condition in self.plan[field] for condition in conditions}

            def step(self, action):
                self.steps[self.episode_index] += 1
                touched = [condition.split()[1] for condition in self.plan["touches"]]
                self.sim = contact_sim(subjects, touched)
                if self.plan.get("touch_error") and self.steps[self.episode_index] > wait_steps:
                    self.sim.data = None
                done = any(self.evaluate_conditions(self.conditions).values())
                return self.observation(), 0.0, done, {}

            def infer(self, observation):
                self.infers[self.episode_index] += 1
                if observation["prompt"] != "pick up the milk and place it in the basket":
                    raise AssertionError("Changed client prompt")
                if self.plan.get("infer_error"):
                    raise RuntimeError("inference failed")
                return {"actions": np.zeros((1, 7))}

            def close(self):
                self.closed = True

        env = Environment()
        task = SimpleNamespace(
            problem_folder=suite_name, bddl_file=task_key.split("/", 1)[1],
            language="pick up the milk and place it in the basket",
        )
        suite = SimpleNamespace(
            name=suite_name, n_tasks=1, get_task=lambda _: task,
            get_task_init_states=lambda _: list(range(len(plans))), get_task_max_steps=lambda _: 2,
        )
        client = SimpleNamespace(
            get_server_metadata=lambda: {
                "input_modality": "language", "mask_language_tokens": False,
                "mask_vision_tokens": True, "model_backend": "jax",
            },
            infer=env.infer,
        )
        media = Mock(side_effect=RuntimeError("video encoder unavailable") if media_error else None)
        namespace.update({
            "benchmark": SimpleNamespace(get_benchmark_dict=lambda: {suite_name: lambda: suite}),
            "_websocket_client_policy": SimpleNamespace(WebsocketClientPolicy=lambda *_: client),
            "datetime": SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: "timestamp")),
            "tqdm": SimpleNamespace(tqdm=lambda value: value),
            "get_libero_path": lambda _: str(CONDITIONS_DIR),
            "_load_custom_conditions_db": lambda _: db,
            "_get_libero_env": lambda *_: (env, task.language),
            "_setup_logging": Mock(), "_save_points_as_ply": Mock(),
            "_quat2axisangle": lambda _: np.zeros(3),
            "image_tools": SimpleNamespace(convert_to_uint8=lambda image: image, resize_with_pad=lambda image, *_: image),
            "imageio": SimpleNamespace(mimwrite=media),
        })
        args = namespace["Args"](
            task_suite_name=suite_name, num_trials_per_task=len(plans), num_steps_wait=wait_steps,
            replan_steps=1, stop_on_error=stop_on_error,
        )
        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary_dir:
            try:
                os.chdir(temporary_dir)
                with contextlib.redirect_stdout(io.StringIO()):
                    if stop_on_error:
                        with self.assertRaisesRegex(RuntimeError, "failed"):
                            namespace["eval_libero"](args)
                    else:
                        namespace["eval_libero"](args)
                output = Path("rollouts/{}-language-timestamp".format(suite_name))
                episodes = sorted(
                    [read_json(path) for path in (output / "trajectories").glob("*.json")],
                    key=lambda result: result["episode_index"],
                )
                summary = read_json(output / "metrics_summary.json")
                catalog = read_json(output / "evaluation_catalog.json")
                self.assertEqual(catalog["tasks"][task_key], spec)
            finally:
                os.chdir(previous_directory)
        return episodes, summary, env, roles, media

    def test_real_loop_handles_multiple_episodes_faults_and_unknown_touch(self):
        episodes, summary, env, roles, _ = self.run_scenarios([
            {"successes": ["other"], "touches": ["original", "other"]},
            {"reset_error": True},
            {"successes": ["following", "other"], "touch_error": True},
            {},
            {"initial": ["other"]},
            {"infer_error": True},
        ])
        self.assertEqual([result["outcome"] for result in episodes], [
            "deviation_only", "error", "both", "failure", "invalid_initial", "error",
        ])
        self.assertEqual([result["end_reason"] for result in episodes], [
            "candidate_success", "error", "candidate_success", "time_limit", "invalid_initial", "error",
        ])
        self.assertEqual(episodes[0]["deviation"]["by_goal"][roles["other"]]["success"], True)
        self.assertFalse(episodes[0]["deviation"]["original_training"]["success"])
        self.assertEqual(episodes[0]["result"], "success")
        self.assertEqual(episodes[1]["result"], "error")
        self.assertIsNone(episodes[1]["following"]["success"])
        self.assertFalse(episodes[2]["touch_valid"])
        self.assertIsNone(episodes[2]["following"]["touched"])
        self.assertTrue(episodes[2]["following"]["success"])
        self.assertEqual(episodes[3]["result"], "failure")
        self.assertEqual(len(episodes[3]["steps"]), 2)
        self.assertEqual(env.infers[1], 0)
        self.assertEqual(env.infers[4], 0)
        self.assertEqual(env.steps[4], 1)  # Stabilization only, no policy action.
        self.assertTrue(env.closed)
        self.assertEqual(summary["pooled"]["counts"], {
            "attempted": 6, "valid": 3, "error": 2, "invalid_initial": 1, "touch_valid": 2,
        })
        self.assertAlmostEqual(summary["pooled"]["following"]["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["pooled"]["deviation"]["success_rate"], 2 / 3)
        self.assertEqual(summary["pooled"]["deviation"]["touch_rate_sum"], 1.0)

    def test_stop_on_error_saves_episode_and_summary_before_raising(self):
        episodes, summary, env, _, _ = self.run_scenarios([{"infer_error": True}], stop_on_error=True)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["status"], "error")
        self.assertEqual(summary["pooled"]["counts"]["error"], 1)
        self.assertIsNone(summary["pooled"]["following"]["success_rate"])
        self.assertTrue(env.closed)

    def test_video_export_failure_preserves_scoring_and_later_episodes(self):
        episodes, summary, _, _, media = self.run_scenarios([
            {"successes": ["following"]}, {},
        ], media_error=True)
        self.assertEqual([result["outcome"] for result in episodes], ["following_only", "failure"])
        self.assertEqual(summary["pooled"]["counts"]["valid"], 2)
        self.assertEqual(summary["pooled"]["following"]["success_rate"], 0.5)
        self.assertEqual(media.call_count, 2)


if __name__ == "__main__":
    unittest.main()
