"""Following/deviation metrics for the approved same-destination protocol.

This module deliberately has no simulator or numerical-library dependencies.
Rates are fractions. Success uses valid episodes; touch uses valid episodes
whose contact observations are complete. A deviation touch-rate sum is a sum
over distinct objects, not the probability of touching any deviation object.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SUPPORTED_SUITES = frozenset(("libero_cf_object", "libero_cf_spatial", "libero_cf_ood"))
PROTOCOL = "same_destination_first_success_v1"
RATE_UNIT = "fraction"
_VALID_END_REASONS = frozenset(("candidate_success", "time_limit"))
_END_REASONS = _VALID_END_REASONS | frozenset(("error", "invalid_initial"))
_OUTCOMES = ("following_only", "deviation_only", "both", "failure", "error", "invalid_initial")
_COUNT_KEYS = ("attempted", "valid", "error", "invalid_initial", "touch_valid")


def _validate_spec(spec: Mapping[str, Any], context: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("{}: task metadata must be an object".format(context))
    conditions = spec.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 2:
        raise ValueError("{}: conditions must contain at least two entries".format(context))
    if any(not isinstance(c, str) for c in conditions):
        raise ValueError("{}: conditions must be strings".format(context))
    if len(set(conditions)) != len(conditions):
        raise ValueError("{}: duplicate conditions".format(context))
    subjects = []
    destinations = set()
    for condition in conditions:
        tokens = condition.split()
        if len(tokens) != 3 or tokens[0] not in ("in", "on") or " ".join(tokens) != condition:
            raise ValueError("{}: invalid placement condition {!r}".format(context, condition))
        predicate, subject, target = tokens
        if target == subject or target.startswith(subject + "_"):
            raise ValueError("{}: a receiving object cannot be placed in/on itself".format(context))
        subjects.append(subject)
        destinations.add((predicate, target))
    if len(set(subjects)) != len(subjects):
        raise ValueError("{}: each object must have exactly one condition".format(context))
    if len(destinations) != 1:
        raise ValueError("{}: this protocol requires one placement predicate/destination".format(context))
    following = spec.get("following_condition")
    if not isinstance(following, str) or following not in conditions:
        raise ValueError("{}: following_condition must be an explicit candidate".format(context))
    original = spec.get("original_training")
    if not isinstance(original, dict):
        raise ValueError("{}: missing original_training mapping".format(context))
    for key in ("suite", "task", "condition"):
        if not isinstance(original.get(key), str) or not original[key]:
            raise ValueError("{}: invalid original_training.{}".format(context, key))
    if original["condition"] not in conditions or original["condition"] == following:
        raise ValueError("{}: original condition must be a distinct deviation candidate".format(context))
    return copy.deepcopy(spec)


def load_task_metadata(conditions_dir: Path, suite: str, conditions_db: dict) -> Dict[str, Any]:
    """Load explicit task roles and reject stale, incomplete or ambiguous data."""
    if suite not in SUPPORTED_SUITES:
        raise ValueError("Unsupported outcome-metrics suite: {}".format(suite))
    if not isinstance(conditions_db, dict):
        raise ValueError("conditions_db must be a task-to-conditions mapping")
    path = Path(conditions_dir) / "metadata" / (suite + ".json")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("{}: metadata must be an object".format(path))
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("{}: unsupported schema_version".format(path))
    if metadata.get("suite") != suite:
        raise ValueError("{}: suite does not match {}".format(path, suite))
    if metadata.get("scope") != "same_destination_pick_and_place":
        raise ValueError("{}: unsupported metadata scope".format(path))
    tasks = metadata.get("tasks")
    if not isinstance(tasks, dict) or not tasks or set(tasks) != set(conditions_db):
        raise ValueError("{}: metadata and conditions_db must contain the same complete task keys".format(path))
    validated = {}
    for task_key, spec in tasks.items():
        if not isinstance(task_key, str) or not task_key.startswith(suite + "/") or not task_key.endswith(".bddl"):
            raise ValueError("{}: invalid task key {!r}".format(path, task_key))
        parsed = _validate_spec(spec, task_key)
        if not isinstance(conditions_db[task_key], list) or parsed["conditions"] != conditions_db[task_key]:
            raise ValueError("{}: metadata conditions differ from conditions_db, including order".format(task_key))
        expected_original_suite = "libero_object" if suite == "libero_cf_object" else "libero_spatial"
        if parsed["original_training"]["suite"] != expected_original_suite:
            raise ValueError("{}: unexpected original benchmark suite".format(task_key))
        validated[task_key] = parsed
    return validated


def _boolean_at(values: Mapping[str, Any], condition: str, label: str) -> bool:
    if not isinstance(values, Mapping) or condition not in values or type(values[condition]) is not bool:
        raise ValueError("{} must contain a boolean for {!r}".format(label, condition))
    return values[condition]


def _goal_identity(spec: Mapping[str, Any], condition: str) -> Dict[str, Any]:
    return {
        "condition": condition,
        "role": "following" if condition == spec["following_condition"] else "deviation",
        "is_original_training_goal": condition == spec["original_training"]["condition"],
        "subject": condition.split()[1],
    }


def build_episode_result(
    spec: Mapping[str, Any],
    condition_results: Mapping[str, Any],
    touched_by_condition: Mapping[str, Any],
    *,
    end_reason: str,
    error: Optional[Any] = None,
    touch_valid: bool = True,
) -> Dict[str, Any]:
    """Classify one episode without turning unknown/error observations into failures.

    For valid episodes all condition results must be present and boolean. When
    contact collection is incomplete, pass touch_valid=False; all touch fields
    then become null, while success observations remain usable.
    """
    spec = _validate_spec(spec, "episode")
    if end_reason not in _END_REASONS:
        raise ValueError("Unsupported end_reason: {}".format(end_reason))
    if type(touch_valid) is not bool:
        raise ValueError("touch_valid must be boolean")
    valid = end_reason in _VALID_END_REASONS
    contact_valid = valid and touch_valid
    per_condition = []
    for condition in spec["conditions"]:
        goal = _goal_identity(spec, condition)
        goal["success"] = _boolean_at(condition_results, condition, "condition_results") if valid else None
        goal["touched"] = _boolean_at(touched_by_condition, condition, "touched_by_condition") if contact_valid else None
        per_condition.append(goal)
    if valid:
        any_success = any(goal["success"] for goal in per_condition)
        if any_success != (end_reason == "candidate_success"):
            raise ValueError("end_reason does not match observed candidate conditions")
    by_condition = {goal["condition"]: goal for goal in per_condition}
    following = copy.deepcopy(by_condition[spec["following_condition"]])
    deviations = {c: copy.deepcopy(g) for c, g in by_condition.items() if c != spec["following_condition"]}
    successful = [c for c, goal in deviations.items() if goal["success"]] if valid else None
    deviation_success = bool(successful) if valid else None
    if not valid:
        outcome = end_reason
    elif following["success"] and deviation_success:
        outcome = "both"
    elif following["success"]:
        outcome = "following_only"
    elif deviation_success:
        outcome = "deviation_only"
    else:
        outcome = "failure"
    original = copy.deepcopy(spec["original_training"])
    original.update(copy.deepcopy(deviations[original["condition"]]))
    return {
        "status": "valid" if valid else end_reason,
        "end_reason": end_reason,
        "error": str(error) if error is not None else None,
        "touch_valid": contact_valid,
        "outcome": outcome,
        "following": following,
        "deviation": {
            "success": deviation_success,
            "successful_conditions": successful,
            "original_training": original,
            "by_goal": deviations,
        },
        "per_condition": per_condition,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


class TaskMetrics:
    """Accumulate counts; retain no trajectory or episode objects."""

    def __init__(self, task_key: str, spec: Mapping[str, Any]) -> None:
        self.task_key = task_key
        self.spec = _validate_spec(spec, task_key)
        self._counts = {key: 0 for key in _COUNT_KEYS}
        self._outcomes = {key: 0 for key in _OUTCOMES}
        self._successes = {condition: 0 for condition in self.spec["conditions"]}
        self._touches = {condition: 0 for condition in self.spec["conditions"]}

    def add_episode(self, result: Mapping[str, Any]) -> None:
        goals = result.get("per_condition")
        if not isinstance(goals, list) or [g.get("condition") for g in goals] != self.spec["conditions"]:
            raise ValueError("{}: episode conditions do not match this task".format(self.task_key))
        # Rebuild from observations so inconsistent labels cannot corrupt counts.
        checked = build_episode_result(
            self.spec,
            {g["condition"]: g.get("success") for g in goals},
            {g["condition"]: g.get("touched") for g in goals},
            end_reason=result.get("end_reason"),
            error=result.get("error"),
            touch_valid=result.get("touch_valid", False),
        )
        if result.get("status") != checked["status"] or result.get("outcome") != checked["outcome"]:
            raise ValueError("{}: inconsistent episode status/outcome".format(self.task_key))
        self._counts["attempted"] += 1
        self._outcomes[checked["outcome"]] += 1
        if checked["status"] != "valid":
            self._counts[checked["status"]] += 1
            return
        self._counts["valid"] += 1
        if checked["touch_valid"]:
            self._counts["touch_valid"] += 1
        for goal in checked["per_condition"]:
            self._successes[goal["condition"]] += int(goal["success"])
            if checked["touch_valid"]:
                self._touches[goal["condition"]] += int(goal["touched"])

    def _goal_summary(self, condition: str) -> Dict[str, Any]:
        result = _goal_identity(self.spec, condition)
        result.update({
            "successes": self._successes[condition],
            "success_rate": _rate(self._successes[condition], self._counts["valid"]),
            "touches": self._touches[condition],
            "touch_rate": _rate(self._touches[condition], self._counts["touch_valid"]),
        })
        return result

    def summary(self) -> Dict[str, Any]:
        following = self._goal_summary(self.spec["following_condition"])
        deviations = {c: self._goal_summary(c) for c in self.spec["conditions"] if c != self.spec["following_condition"]}
        original = copy.deepcopy(self.spec["original_training"])
        original.update(copy.deepcopy(deviations[original["condition"]]))
        only = self._outcomes["deviation_only"]
        any_deviation = only + self._outcomes["both"]
        touches_sum = sum(goal["touches"] for goal in deviations.values())
        return {
            "task_key": self.task_key,
            "protocol": PROTOCOL,
            "rate_unit": RATE_UNIT,
            "counts": dict(self._counts),
            "outcomes": dict(self._outcomes),
            "following": following,
            "deviation": {
                "successes": any_deviation,
                "success_rate": _rate(any_deviation, self._counts["valid"]),
                "only_successes": only,
                "only_success_rate": _rate(only, self._counts["valid"]),
                "touches_sum": touches_sum,
                "touch_rate_sum": sum(g["touch_rate"] for g in deviations.values()) if self._counts["touch_valid"] else None,
                "original_training": original,
                "by_goal": deviations,
            },
        }


def _mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize_run(suite: str, task_summaries: dict) -> Dict[str, Any]:
    """Return per-task detail, pooled episode rates and equal-task macro rates.

    Individual goal predicates are deliberately not merged across scenes. They
    remain in tasks[task_key], even if several tasks use the same condition text.
    """
    if suite not in SUPPORTED_SUITES:
        raise ValueError("Unsupported outcome-metrics suite: {}".format(suite))
    if not isinstance(task_summaries, dict):
        raise ValueError("task_summaries must be a task-keyed mapping")
    tasks = copy.deepcopy(task_summaries)
    for task_key, task in tasks.items():
        if task.get("task_key") != task_key or not task_key.startswith(suite + "/"):
            raise ValueError("Task summary key/suite mismatch: {}".format(task_key))
        if task.get("protocol") != PROTOCOL or task.get("rate_unit") != RATE_UNIT:
            raise ValueError("Incompatible task summary protocol/unit: {}".format(task_key))
    items = list(tasks.values())
    counts = {key: sum(t["counts"][key] for t in items) for key in _COUNT_KEYS}
    outcomes = {key: sum(t["outcomes"][key] for t in items) for key in _OUTCOMES}
    success_tasks = [t for t in items if t["counts"]["valid"] > 0]
    touch_tasks = [t for t in items if t["counts"]["touch_valid"] > 0]

    def pooled_goal(goals: list) -> Dict[str, Any]:
        successes = sum(g["successes"] for g in goals)
        touches = sum(g["touches"] for g in goals)
        return {
            "successes": successes,
            "success_rate": _rate(successes, counts["valid"]),
            "touches": touches,
            "touch_rate": _rate(touches, counts["touch_valid"]),
        }

    any_deviation = sum(t["deviation"]["successes"] for t in items)
    only_deviation = sum(t["deviation"]["only_successes"] for t in items)
    touches_sum = sum(t["deviation"]["touches_sum"] for t in items)
    pooled = {
        "counts": counts,
        "outcomes": outcomes,
        "following": pooled_goal([t["following"] for t in items]),
        "deviation": {
            "successes": any_deviation,
            "success_rate": _rate(any_deviation, counts["valid"]),
            "only_successes": only_deviation,
            "only_success_rate": _rate(only_deviation, counts["valid"]),
            "touches_sum": touches_sum,
            "touch_rate_sum": _rate(touches_sum, counts["touch_valid"]),
            "original_training": pooled_goal([t["deviation"]["original_training"] for t in items]),
        },
    }
    macro = {
        "valid_task_counts": {"success": len(success_tasks), "touch": len(touch_tasks)},
        "following": {
            "success_rate": _mean([t["following"]["success_rate"] for t in success_tasks]),
            "touch_rate": _mean([t["following"]["touch_rate"] for t in touch_tasks]),
        },
        "deviation": {
            "success_rate": _mean([t["deviation"]["success_rate"] for t in success_tasks]),
            "only_success_rate": _mean([t["deviation"]["only_success_rate"] for t in success_tasks]),
            "touch_rate_sum": _mean([t["deviation"]["touch_rate_sum"] for t in touch_tasks]),
            "original_training": {
                "success_rate": _mean([t["deviation"]["original_training"]["success_rate"] for t in success_tasks]),
                "touch_rate": _mean([t["deviation"]["original_training"]["touch_rate"] for t in touch_tasks]),
            },
        },
    }
    return {
        "suite": suite,
        "protocol": PROTOCOL,
        "rate_unit": RATE_UNIT,
        "task_count": len(tasks),
        "denominators": {"success": "valid episodes", "touch": "valid episodes with complete touch observations"},
        "tasks": tasks,
        "pooled": pooled,
        "macro": macro,
    }
