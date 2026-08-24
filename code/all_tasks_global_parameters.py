from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


TIERS = ("T0", "T1", "T2", "T3")
SPLITS = ("model_train", "mdp_build", "post_cal", "report_test")
ACTIONS = ("NEG", "POS", "REVIEW", "CONTINUE")


@dataclass(frozen=True)
class Contract:
                                                                    

    contract_id: str
    evidence_schedule: str
    action_profile: str
    tier_costs: dict[str, float]
    false_positive_price: float
    review_price: float

    @property
    def positive_threshold(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.review_price / self.false_positive_price))

    @property
    def price_vector(self) -> tuple[float, float, float, float, float]:
        return (
            self.tier_costs["T1"] - self.tier_costs["T0"],
            self.tier_costs["T2"] - self.tier_costs["T1"],
            self.tier_costs["T3"] - self.tier_costs["T2"],
            self.false_positive_price,
            self.review_price,
        )


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    raw: dict[str, Any]

    @property
    def task_key(self) -> str:
        return str(self.raw["task_key"])

    @property
    def task_name(self) -> str:
        return str(self.raw["task_name"])

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw["data"])

    @property
    def split(self) -> dict[str, Any]:
        return dict(self.raw["split"])

    @property
    def incremental_features(self) -> dict[str, list[str]]:
        return {tier: list(self.raw["features"][tier]) for tier in TIERS}

    @property
    def cumulative_features(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        seen: list[str] = []
        for tier in TIERS:
            seen.extend(self.raw["features"][tier])
            out[tier] = list(dict.fromkeys(seen))
        return out


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_global_parameters(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    errors = validate_global_parameters(config)
    if errors:
        raise ValueError("invalid global DASR configuration:\n- " + "\n- ".join(errors))
    return config


def load_task_spec(path: str | Path) -> TaskSpec:
    resolved = Path(path).resolve()
    spec = TaskSpec(resolved, read_json(resolved))
    errors = validate_task_spec(spec)
    if errors:
        raise ValueError(f"invalid task specification {resolved}:\n- " + "\n- ".join(errors))
    return spec


def load_task_directory(path: str | Path) -> tuple[dict[str, Any], list[TaskSpec]]:
    root = Path(path).resolve()
    global_config = load_global_parameters(root / "global_parameters.json")
    tasks = [load_task_spec(root / f"{key}.json") for key in ("cfpb", "inspire", "mds")]
    return global_config, tasks


def registered_contracts(config: dict[str, Any]) -> list[Contract]:
    schedules = {row["id"]: row for row in config["evidence_schedules"]}
    profiles = {row["id"]: row for row in config["action_profiles"]}
    out: list[Contract] = []
    for schedule_id, schedule in schedules.items():
        costs = {tier: float(schedule["cumulative_costs"][tier]) for tier in TIERS}
        for profile_id, profile in profiles.items():
            out.append(Contract(
                contract_id=f"{schedule_id}__{profile_id}",
                evidence_schedule=schedule_id,
                action_profile=profile_id,
                tier_costs=costs,
                false_positive_price=float(profile["false_positive_price"]),
                review_price=float(profile["review_price"]),
            ))
    return out


def contract_by_id(config: dict[str, Any], contract_id: str) -> Contract:
    matches = [contract for contract in registered_contracts(config)
               if contract.contract_id == contract_id]
    if len(matches) != 1:
        known = ", ".join(item.contract_id for item in registered_contracts(config))
        raise KeyError(f"unknown contract {contract_id!r}; choose one of: {known}")
    return matches[0]


def validate_global_parameters(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "dasr-global-v1":
        errors.append("schema_version must be dasr-global-v1")
    if tuple(config.get("split_roles", [])) != SPLITS:
        errors.append(f"split_roles must be {list(SPLITS)} in this order")
    fractions = config.get("split_fractions", {})
    if set(fractions) != set(SPLITS) or abs(sum(float(v) for v in fractions.values()) - 1.0) > 1e-12:
        errors.append("split_fractions must name all four roles and sum to one")
    audit = config.get("audit", {})
    bins = int(audit.get("score_bins_per_tier", 0))
    if int(audit.get("slice_count", 0)) != len(TIERS) * bins:
        errors.append("audit.slice_count must equal four tiers times score bins")
    if not 0.0 < float(audit.get("selection_cap", 0.0)) < float(audit.get("report_cap", 0.0)) < 1.0:
        errors.append("audit caps must satisfy 0 < selection_cap < report_cap < 1")
    if not 0.0 < float(audit.get("tail_mass", 0.0)) <= 1.0:
        errors.append("audit.tail_mass must lie in (0,1]")
    schedules = config.get("evidence_schedules", [])
    profiles = config.get("action_profiles", [])
    if len(schedules) != 3 or len(profiles) != 3:
        errors.append("the registered operating domain must contain three schedules and three profiles")
    for row in schedules:
        costs = row.get("cumulative_costs", {})
        if set(costs) != set(TIERS):
            errors.append(f"schedule {row.get('id')} must define T0--T3")
            continue
        values = [float(costs[tier]) for tier in TIERS]
        if any(right < left for left, right in zip(values, values[1:])):
            errors.append(f"schedule {row.get('id')} must be cumulative and nondecreasing")
        if abs(values[0]) > 1e-12 or abs(values[-1] - 11.1) > 1e-6:
            errors.append(f"schedule {row.get('id')} must start at 0 and end at 11.1")
    for row in profiles:
        if float(row.get("false_positive_price", 0.0)) <= 0 or float(row.get("review_price", 0.0)) <= 0:
            errors.append(f"profile {row.get('id')} prices must be positive")
    policy_classes = config.get("structure", {}).get("policy_classes", [])
    if policy_classes != ["score_anchor", "path_free", "path_bearing"]:
        errors.append("structure.policy_classes must preserve the final unified class order")
    return errors


def validate_task_spec(spec: TaskSpec) -> list[str]:
    raw = spec.raw
    errors: list[str] = []
    if raw.get("schema_version") != "dasr-task-v1":
        errors.append("schema_version must be dasr-task-v1")
    for key in ("task_key", "paper_name", "task_name", "data", "split", "features", "expected"):
        if key not in raw:
            errors.append(f"missing required field: {key}")
    if errors:
        return errors
    if set(raw["features"]) != set(TIERS):
        errors.append("features must contain exactly T0, T1, T2, and T3")
    all_features = [name for tier in TIERS for name in raw["features"].get(tier, [])]
    if len(all_features) != len(set(all_features)):
        errors.append("incremental feature lists must not repeat a feature across tiers")
    expected_counts = raw["expected"].get("cumulative_feature_counts", {})
    actual_counts = {tier: len(features) for tier, features in spec.cumulative_features.items()}
    if expected_counts != actual_counts:
        errors.append(f"cumulative feature counts {actual_counts} do not match expected {expected_counts}")
    data = raw["data"]
    for key in ("path", "format", "target_col", "row_id_col"):
        if not data.get(key):
            errors.append(f"data.{key} is required")
    if data.get("format") not in {"csv", "parquet"}:
        errors.append("data.format must be csv or parquet")
    if raw["split"].get("strategy") not in {
        "deterministic_stratified_row", "deterministic_group_hash",
        "deterministic_stratified_group"
    }:
        errors.append("unsupported split.strategy")
    if "group" in str(raw["split"].get("strategy")) and not data.get("group_id_col"):
        errors.append("group split strategies require data.group_id_col")
    selection_rule = raw["split"].get("group_row_selection", "stable_hash")
    if selection_rule not in {"stable_hash", "earliest_numeric_then_row_id"}:
        errors.append("unsupported split.group_row_selection")
    if (selection_rule == "earliest_numeric_then_row_id"
            and not data.get("audit_order_col")):
        errors.append(
            "earliest_numeric_then_row_id requires data.audit_order_col")
    return errors


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def require_fields(mapping: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise KeyError(f"{context} is missing fields: {missing}")
