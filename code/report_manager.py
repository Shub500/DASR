                                                                     

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    return target


def write_contract_report(path: str | Path, summary: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selection = summary["selection"]
    audit = summary["sealed_audit"]
    lines = [
        f"# DASR contract report: {summary['contract_id']}",
        "",
        f"- Task: `{summary['task_key']}`",
        "- Implementation: Standalone DASR reference implementation",
        f"- Selected class: `{selection['family']}`",
        f"- Selected policy: `{selection['policy_hash']}`",
        f"- Policy Selection cost: {selection['cost']:.6f}",
        f"- Policy Selection tail UCB: {selection['tail_risk_ucb']:.6f} (cap {selection['checked_cap']:.3f})",
        f"- Sealed Audit cost: {audit['cost']:.6f}",
        f"- Sealed Audit tail UCB: {audit['tail_risk_ucb']:.6f} (cap {audit['checked_cap']:.3f})",
        f"- Sealed Audit pass: **{audit['safety_pass']}**",
        f"- Sealed-audit accesses: {len(summary['sealed_audit_access_log'])}",
        "",
        "## Consequence card",
        "",
        "`(reach T1, reach T2, reach T3, false-positive mass, review mass)`",
        "",
        f"`{audit['consequence_vector']}`",
        "",
        "The direct case-cost mean and exact linear repricing from this card are checked for equality.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_run_index(path: str | Path, task_key: str,
                    contract_summaries: list[dict[str, Any]]) -> Path:
    target = Path(path)
    lines = [f"# DASR standalone reference run: {task_key}", "", "| Contract | Class | Post cost | Audit UCB | Pass |", "|---|---|---:|---:|---:|"]
    for summary in contract_summaries:
        selection, audit = summary["selection"], summary["sealed_audit"]
        lines.append(
            f"| {summary['contract_id']} | {selection['family']} | {selection['cost']:.6f} | "
            f"{audit['tail_risk_ucb']:.6f} | {audit['safety_pass']} |")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
