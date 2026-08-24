                                                                   

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from all_tasks_global_parameters import SPLITS, TaskSpec


def resolve_data_path(spec: TaskSpec, data_root: str | Path | None = None,
                      data_path: str | Path | None = None) -> Path:
    if data_path is not None:
        return Path(data_path).expanduser().resolve()
    relative = Path(str(spec.data["path"]))
    root = Path(data_root).expanduser().resolve() if data_root else spec.path.parent
    return (root / relative).resolve()


def read_prepared_data(path: str | Path, file_format: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"prepared dataset not found: {source}")
    if file_format == "parquet":
        return pd.read_parquet(source)
    if file_format == "csv":
        return pd.read_csv(source, low_memory=False)
    raise ValueError(f"unsupported data format: {file_format}")


def validate_dataset_schema(path: str | Path, spec: TaskSpec) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"status": "missing", "path": str(source), "missing_columns": []}
    if spec.data["format"] == "parquet":
        import pyarrow.parquet as pq
        columns = set(pq.ParquetFile(source).schema.names)
    else:
        columns = set(pd.read_csv(source, nrows=0).columns)
    required = dataset_columns(spec)
    missing = sorted(required - columns)
    return {
        "status": "pass" if not missing else "fail",
        "path": str(source),
        "column_count": len(columns),
        "required_column_count": len(required),
        "missing_columns": missing,
    }


def dataset_columns(spec: TaskSpec) -> set[str]:
    data = spec.data
    columns = {str(data["target_col"]), str(data["row_id_col"])}
    columns.update(name for names in spec.incremental_features.values() for name in names)
    for optional in (data.get("group_id_col"), data.get("audit_order_col")):
        if optional:
            columns.add(str(optional))
    return columns


def load_task_frame(spec: TaskSpec, global_config: dict[str, Any],
                    data_root: str | Path | None = None,
                    data_path: str | Path | None = None) -> pd.DataFrame:
    source = resolve_data_path(spec, data_root, data_path)
    frame = read_prepared_data(source, str(spec.data["format"]))
    frame = prepare_frame(frame, spec)
    frame = apply_split_contract(frame, spec, global_config)
    return frame.reset_index(drop=True)


def prepare_frame(frame: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    missing = sorted(dataset_columns(spec) - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.task_key} prepared data is missing columns: {missing}")
    data = spec.data
    out = frame.copy()
    target_col = str(data["target_col"])
    invalid = set(data.get("invalid_target_values", []))
    if invalid:
        out = out.loc[~out[target_col].isin(invalid)].copy()
    target = pd.to_numeric(out[target_col], errors="coerce")
    out = out.loc[target.notna()].copy()
    target = target.loc[target.notna()].astype(int)
    if not set(target.unique()) <= {0, 1}:
        raise ValueError(f"{spec.task_key} target must be binary after invalid values are removed")
    out["target"] = target.to_numpy(int)
    out["row_id"] = out[str(data["row_id_col"])].astype(str)
    group_col = data.get("group_id_col")
    out["audit_group_id"] = out[str(group_col)].astype(str) if group_col else out["row_id"]
    out["_source_order"] = np.arange(len(out), dtype=np.int64)
    if bool(spec.split.get("deduplicate_one_row_per_group", False)):
        out = select_one_row_per_group(out, spec, salt="deduplicate")
    return out.reset_index(drop=True)


def apply_split_contract(frame: pd.DataFrame, spec: TaskSpec,
                         global_config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    if "split" in out and set(out["split"].dropna().astype(str).unique()) <= set(SPLITS):
        out["split"] = out["split"].astype(str)
    else:
        out["split"] = generated_splits(out, spec, global_config)
    if set(out["split"].unique()) != set(SPLITS):
        raise ValueError(f"all four split roles are required; found {sorted(out['split'].unique())}")
    if bool(spec.split.get("audit_one_row_per_group", False)):
        development = out.loc[out["split"].isin(["model_train", "mdp_build"])]
        audit = out.loc[out["split"].isin(["post_cal", "report_test"])]
        audit = select_one_row_per_group(audit, spec, salt="audit")
        out = pd.concat([development, audit], ignore_index=True)
    assert_group_isolation(out)
    return out.reset_index(drop=True)


def generated_splits(frame: pd.DataFrame, spec: TaskSpec,
                     global_config: dict[str, Any]) -> np.ndarray:
    strategy = str(spec.split["strategy"])
    seed = int(global_config["random_seed"])
    fractions = {key: float(global_config["split_fractions"][key]) for key in SPLITS}
    if strategy == "deterministic_stratified_row":
        units = frame[["row_id", "target"]].rename(columns={"row_id": "unit"}).copy()
        assigned = assign_units(units, fractions, seed, stratified=True)
        return frame["row_id"].map(assigned).to_numpy(object)
    group = frame.groupby("audit_group_id", sort=False)["target"].max().rename("target").reset_index()
    group = group.rename(columns={"audit_group_id": "unit"})
    assigned = assign_units(
        group, fractions, seed,
        stratified=(strategy == "deterministic_stratified_group"),
    )
    return frame["audit_group_id"].map(assigned).to_numpy(object)


def assign_units(units: pd.DataFrame, fractions: dict[str, float], seed: int,
                 stratified: bool) -> dict[str, str]:
    assignment: dict[str, str] = {}
    groups = units.groupby("target", sort=True) if stratified else [("all", units)]
    for _, part in groups:
        ranked = part.assign(_hash=[stable_uint64(value, seed, "split") for value in part["unit"]])
        ranked = ranked.sort_values(["_hash", "unit"], kind="mergesort")
        counts = fractional_counts(len(ranked), fractions)
        start = 0
        for role in SPLITS:
            stop = start + counts[role]
            assignment.update({str(unit): role for unit in ranked.iloc[start:stop]["unit"]})
            start = stop
    return assignment


def fractional_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {role: total * fractions[role] for role in SPLITS}
    counts = {role: int(np.floor(raw[role])) for role in SPLITS}
    remaining = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda role: (raw[role] - counts[role], role), reverse=True)
    for role in order[:remaining]:
        counts[role] += 1
    return counts


def select_one_row_per_group(frame: pd.DataFrame, spec: TaskSpec, salt: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    order_col = spec.data.get("audit_order_col")
    selection_rule = str(spec.split.get("group_row_selection", "stable_hash"))
    if selection_rule == "earliest_numeric_then_row_id":
        if not order_col or str(order_col) not in out:
            raise ValueError(
                f"{spec.task_key} requires data.audit_order_col for {selection_rule}")
        out["_selection_order"] = pd.to_numeric(out[str(order_col)], errors="coerce")
        out["_selection_order_missing"] = out["_selection_order"].isna()
        out = out.sort_values(
            ["audit_group_id", "_selection_order_missing", "_selection_order",
             "row_id", "_source_order"],
            kind="mergesort",
        )
        return out.drop_duplicates("audit_group_id", keep="first").drop(
            columns=["_selection_order", "_selection_order_missing"])
    if selection_rule != "stable_hash":
        raise ValueError(f"unsupported split.group_row_selection: {selection_rule}")
    if order_col and order_col in out:
        order_text = out[str(order_col)].astype(str)
    else:
        order_text = out["row_id"].astype(str)
    out["_selection_hash"] = [stable_uint64(f"{group}|{order}", 31, salt)
                              for group, order in zip(out["audit_group_id"], order_text)]
    out = out.sort_values(["audit_group_id", "_selection_hash", "_source_order"], kind="mergesort")
    return out.drop_duplicates("audit_group_id", keep="first").drop(columns="_selection_hash")


def assert_group_isolation(frame: pd.DataFrame) -> None:
    counts = frame.groupby("audit_group_id")["split"].nunique()
    if int((counts > 1).sum()):
        raise RuntimeError("group leakage detected across split roles")


def split_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    if role not in SPLITS:
        raise KeyError(f"unknown split role: {role}")
    return frame.loc[frame["split"].eq(role)].copy()


def stable_uint64(value: object, seed: int, salt: str) -> int:
    text = f"{seed}|{salt}|{value}".encode("utf-8")
    return int.from_bytes(sha256(text).digest()[:8], byteorder="big", signed=False)
