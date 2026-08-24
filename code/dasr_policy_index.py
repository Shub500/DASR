                                                                                 

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from all_tasks_global_parameters import content_hash
from dasr_lp_solver import Policy


@dataclass
class IndexedPolicy:
    policy: Policy
    executable_table_hash: str

    def public(self) -> dict[str, Any]:
        return self.policy.public() | {
            "executable_table_hash": self.executable_table_hash,
        }


class PolicyIndex:
                                                                                   

    def __init__(self) -> None:
        self._raw: list[IndexedPolicy] = []
        self._frozen: list[IndexedPolicy] | None = None

    def add(self, policy: Policy) -> None:
        if self._frozen is not None:
            raise RuntimeError("policy index is already frozen")
        if not policy.frozen_before_selection:
            raise ValueError("only construction-frozen policies may enter the index")
        self._raw.append(IndexedPolicy(policy, executable_table_hash(policy)))

    def freeze(self) -> list[IndexedPolicy]:
        if self._frozen is None:
            classes: dict[str, list[IndexedPolicy]] = {}
            for entry in self._raw:
                classes.setdefault(entry.executable_table_hash, []).append(entry)
            representatives = []
            for _table_hash, entries in classes.items():
                chosen = min(entries, key=lambda item: (
                    item.policy.construction_objective,
                    item.policy.representation.state_count(),
                    item.policy.policy_hash,
                ))
                representatives.append(chosen)
            self._frozen = sorted(representatives, key=lambda item: (
                item.policy.family, item.policy.construction_objective, item.policy.policy_hash))
        if not self._frozen:
            raise RuntimeError("cannot freeze an empty policy index")
        return list(self._frozen)

    @property
    def raw_count(self) -> int:
        return len(self._raw)

    @property
    def executable_unique_count(self) -> int:
        return len(self.freeze())

    def manifest(self) -> dict[str, Any]:
        frozen = self.freeze()
        return {
            "implementation": "standalone_reference",
            "frozen_before_policy_selection": True,
            "contains_policy_selection_metrics": False,
            "contains_sealed_audit_metrics": False,
            "policies": [entry.public() for entry in frozen],
        }

    def write_manifest(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def executable_table_hash(policy: Policy) -> str:
                                                                                   
    admissions = [{
        "tier": row.tier,
        "parent_state": row.parent_state,
        "address": row.address,
        "child_state": row.child_state,
        "residual_state": row.residual_state,
    } for row in policy.representation.admissions]
    return content_hash({
        "rule_registry": policy.representation.rule_registry,
        "admissions": admissions,
        "fractions": policy.fractions,
    })
