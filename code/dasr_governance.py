                                                                            

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from all_tasks_global_parameters import Contract, TIERS, content_hash
from base_cascade_calibration import CalibratedCascade
from dasr_lp_solver import Policy, route_policy
from policy_gate import PolicyEvaluation, evaluate_policy
from sealed_audit import SealedAuditReader


@dataclass(frozen=True)
class GovernanceEdit:
    rule_id: str
    version: int
    source_policy_hash: str
    tier: str
    state: str
    replacement_action: str

    def public(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GovernanceResult:
    status: str
    edit: GovernanceEdit
    source_policy_hash: str
    edited_policy_hash: str
    deployed_policy_hash: str
    addressed_n: int
    changed_n: int
    off_target_changed_n: int
    dormant_replay_changed_n: int
    selection: PolicyEvaluation
    audit: PolicyEvaluation | None
    deployed_policy: Policy

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "edit": self.edit.public(),
            "source_policy_hash": self.source_policy_hash,
            "edited_policy_hash": self.edited_policy_hash,
            "deployed_policy_hash": self.deployed_policy_hash,
            "addressed_n": self.addressed_n,
            "changed_n": self.changed_n,
            "off_target_changed_n": self.off_target_changed_n,
            "dormant_replay_changed_n": self.dormant_replay_changed_n,
            "selection": self.selection.public(),
            "audit": self.audit.public() if self.audit else None,
        }


@dataclass(frozen=True)
class MinimaxResult:
    policy_hash: str
    maximum_vertex_regret: float
    vertex_regrets: dict[str, float]

    def public(self) -> dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "maximum_vertex_regret": self.maximum_vertex_regret,
            "vertex_regrets": self.vertex_regrets,
        }


def register_edit(policy: Policy, rule_id: str, version: int,
                  source_policy_hash: str, tier: str, state: str,
                  replacement_action: str) -> GovernanceEdit:
    if not rule_id.strip():
        raise ValueError("rule_id must be nonempty")
    if int(version) < 1:
        raise ValueError("edit version must be positive")
    if source_policy_hash != policy.policy_hash:
        raise ValueError("edit source fingerprint does not match the selected policy")
    if tier not in TIERS:
        raise KeyError(f"unknown tier: {tier}")
    allowed = {"NEG", "POS", "REVIEW"} if tier == TIERS[-1] else {
        "NEG", "POS", "REVIEW", "CONTINUE"}
    if replacement_action not in allowed:
        raise ValueError(f"action {replacement_action} is not permitted at {tier}")
    if state not in policy.fractions.get(tier, {}):
        raise KeyError(f"state {state!r} is not an address in the selected table at {tier}")
    return GovernanceEdit(
        rule_id=rule_id.strip(), version=int(version),
        source_policy_hash=source_policy_hash, tier=tier, state=state,
        replacement_action=replacement_action)


def edit_policy(policy: Policy, edit: GovernanceEdit) -> Policy:
    if edit.source_policy_hash != policy.policy_hash:
        raise ValueError("edit is registered to a different source policy")
    fractions = deepcopy(policy.fractions)
    fractions[edit.tier][edit.state] = {edit.replacement_action: 1.0}
    definition = {
        "family": policy.family,
        "representation_fingerprint": policy.representation.fingerprint,
        "fractions": fractions,
        "governance_edit": edit.public(),
    }
    edited_hash = content_hash(definition)
    return Policy(
        policy_id=f"governance_edit__{edited_hash[:10]}",
        family=policy.family, representation=policy.representation,
        fractions=fractions,
        construction_objective=policy.construction_objective,
        tail_eta=policy.tail_eta, tail_shadow_price=policy.tail_shadow_price,
        flow_residual_max=policy.flow_residual_max, policy_hash=edited_hash,
        flow_values=dict(policy.flow_values), frozen_before_selection=True)


def apply_governance_edit(
        frame: pd.DataFrame, calibrated: CalibratedCascade,
        terminal_action: np.ndarray, selected_policy: Policy,
        contract: Contract, global_config: dict[str, Any], tier: str,
        state: str, replacement_action: str, *, rule_id: str,
        version: int, source_policy_hash: str) -> GovernanceResult:
                                                                                     
    edit = register_edit(
        selected_policy, rule_id, version, source_policy_hash,
        tier, state, replacement_action)
    edited = edit_policy(selected_policy, edit)
    post_mask = frame["split"].eq("post_cal").to_numpy()
    original_routes = route_policy(frame, calibrated, selected_policy, post_mask)
    dormant_routes = route_policy(frame, calibrated, selected_policy, post_mask)
    dormant_changed = int(np.sum(route_change_mask(original_routes, dormant_routes)))
    if dormant_changed:
        raise RuntimeError("source-policy replay did not reproduce before the edit")
    edited_routes = route_policy(frame, calibrated, edited, post_mask)
    addressed = original_routes[f"reached_{tier}"].to_numpy(bool) & (
        original_routes[f"reached_state_{tier}"].astype(str).to_numpy() == state)
    changed = route_change_mask(original_routes, edited_routes)
    off_target = int(np.sum(changed & ~addressed))
    if off_target:
        raise RuntimeError(f"address-local edit changed {off_target} off-target routes")
    selection = evaluate_policy(
        frame, calibrated, terminal_action, edited, contract, global_config,
        split="post_cal", cap=float(global_config["audit"]["selection_cap"]))
    if not selection.safety_pass:
        return governance_result(
            "blocked_at_selection_gate", edit, selected_policy, edited,
            selected_policy, addressed, changed, off_target, dormant_changed,
            selection, None)
    audit_reader = SealedAuditReader(
        frame, calibrated, terminal_action, contract, global_config,
        selected_policy_hash=edited.policy_hash)
    audit = audit_reader.audit(edited)
    audit_reader.assert_single_access()
    if not audit.safety_pass:
        return governance_result(
            "blocked_at_sealed_audit", edit, selected_policy, edited,
            selected_policy, addressed, changed, off_target, dormant_changed,
            selection, audit)
    return governance_result(
        "deployed", edit, selected_policy, edited, edited,
        addressed, changed, off_target, dormant_changed, selection, audit)


def governance_result(status: str, edit: GovernanceEdit, source: Policy,
                      edited: Policy, deployed: Policy, addressed: np.ndarray,
                      changed: np.ndarray, off_target: int,
                      dormant_changed: int, selection: PolicyEvaluation,
                      audit: PolicyEvaluation | None) -> GovernanceResult:
    return GovernanceResult(
        status=status, edit=edit, source_policy_hash=source.policy_hash,
        edited_policy_hash=edited.policy_hash,
        deployed_policy_hash=deployed.policy_hash,
        addressed_n=int(addressed.sum()), changed_n=int(changed.sum()),
        off_target_changed_n=off_target,
        dormant_replay_changed_n=dormant_changed,
        selection=selection, audit=audit, deployed_policy=deployed)


def consequence_cost(consequence: Iterable[float], contract: Contract) -> float:
    values = np.asarray(tuple(consequence), dtype=float)
    if values.shape != (5,):
        raise ValueError("a consequence card must have five coordinates")
    return float(np.dot(np.asarray(contract.price_vector, dtype=float), values))


def minimax_default(evaluations: Iterable[PolicyEvaluation],
                    contracts: Iterable[Contract]) -> MinimaxResult:
                                                                                 
    rows = [row for row in evaluations if row.safety_pass]
    vertices = list(contracts)
    if not rows or not vertices:
        raise ValueError("minimax analysis requires policies and contract vertices")
    cards = {row.policy_hash: row.consequence_vector for row in rows}
    costs = {
        (policy_hash, vertex.contract_id): consequence_cost(card, vertex)
        for policy_hash, card in cards.items() for vertex in vertices}
    best = {
        vertex.contract_id: min(costs[(policy_hash, vertex.contract_id)] for policy_hash in cards)
        for vertex in vertices}
    results: list[MinimaxResult] = []
    for policy_hash in sorted(cards):
        regrets = {
            vertex.contract_id: costs[(policy_hash, vertex.contract_id)] - best[vertex.contract_id]
            for vertex in vertices}
        results.append(MinimaxResult(
            policy_hash=policy_hash,
            maximum_vertex_regret=float(max(regrets.values())),
            vertex_regrets={key: float(value) for key, value in regrets.items()}))
    return min(results, key=lambda row: (row.maximum_vertex_regret, row.policy_hash))


def route_change_mask(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    if not np.array_equal(left["row_id"].to_numpy(object), right["row_id"].to_numpy(object)):
        raise RuntimeError("route comparison row identities are not aligned")
    return ((left["tier"].to_numpy(object) != right["tier"].to_numpy(object)) |
            (left["action"].to_numpy(object) != right["action"].to_numpy(object)))
