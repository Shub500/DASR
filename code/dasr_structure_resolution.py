from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable

import numpy as np
import pandas as pd

from all_tasks_global_parameters import Contract, TIERS, content_hash
from base_cascade_builder import Cascade
from base_cascade_calibration import (
    CalibratedCascade, TerminalReference, allowed_events, largest_negative_threshold,
)


@dataclass
class AdmissionRecord:
    family: str
    proposal_type: str
    tier: str
    parent_state: str
    address: str
    child_state: str
    residual_state: str
    child_n: int
    residual_n: int
    parent_action: str
    child_action: str
    residual_action: str
    action_face_value: float
    cvar_resolution_charge: float
    fixed_dual_gain: float
    exact_objective_before: float | None = None
    exact_objective_after: float | None = None
    exact_route_changed: bool = False

    def public(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Representation:
    representation_id: str
    family: str
    states: dict[str, np.ndarray]
    admissions: list[AdmissionRecord] = field(default_factory=list)
    rule_registry: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def state_count(self, mask: np.ndarray | None = None) -> int:
        return int(sum(len(np.unique(values if mask is None else values[mask]))
                       for values in self.states.values()))

    def public(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "family": self.family,
            "fingerprint": self.fingerprint,
            "state_count": self.state_count(),
            "rule_registry": self.rule_registry,
            "admissions": [row.public() for row in self.admissions],
        }


def score_anchor_representation(calibrated: CalibratedCascade,
                                construction_mask: np.ndarray) -> Representation:
    states = {
        tier: np.asarray([f"{tier}|B={int(value):02d}" for value in calibrated.public_bins[tier]],
                         dtype=object)
        for tier in TIERS
    }
    rules = {
        "kind": "public_calibrated_score_bins",
        "edges": {tier: calibrated.public_bin_edges[tier].tolist() for tier in TIERS},
    }
    return finalize_representation(
        "score_anchor", "score_anchor", states, [], construction_mask, rules)


def compile_structural_representations(
        frame: pd.DataFrame, cascade: Cascade, calibrated: CalibratedCascade,
        terminal_reference: TerminalReference, terminal_action: np.ndarray,
        contract: Contract, global_config: dict[str, Any],
        exact_solver: Callable[[Representation], Any | None],
        route_signature: Callable[[Any], str]) -> list[Representation]:
                                                                                  
    construction = frame["split"].eq("mdp_build").to_numpy()
    zones, boundaries = decision_regions(
        frame, calibrated, construction, contract, global_config)
    families: list[Representation] = []
    for family in ("path_free", "path_bearing"):
        families.extend(search_family(
            family=family, frame=frame, calibrated=calibrated, zones=zones,
            boundaries=boundaries, addresses=address_matrices(cascade, family),
            terminal_reference=terminal_reference, terminal_action=terminal_action,
            contract=contract, global_config=global_config,
            exact_solver=exact_solver,
            route_signature=route_signature, construction_mask=construction))
    return families


def decision_regions(frame: pd.DataFrame, calibrated: CalibratedCascade,
                     construction: np.ndarray, contract: Contract,
                     global_config: dict[str, Any]
                     ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
                                                                               
    labels = frame.loc[construction, "target"].to_numpy(int)
    cfg = global_config["terminal_reference"]
    positive_n = int(np.sum(labels == 1))
    cap = float(cfg["false_negative_cap"]) * float(cfg["construction_reserve"])
    allowed = allowed_events(
        positive_n, cap, float(global_config["audit"]["total_slice_error"]))
    zones: dict[str, np.ndarray] = {}
    boundaries: dict[str, dict[str, float]] = {}
    for tier in TIERS:
        score = calibrated.scores[tier]
        negative, _ = largest_negative_threshold(score[construction], labels, allowed)
        positive = float(contract.positive_threshold)
        region = np.full(len(score), "GRAY", dtype=object)
        region[score <= negative] = "NEG"
        region[score >= positive] = "POS"
        zones[tier] = region
        boundaries[tier] = {"negative": float(negative), "positive": positive}
    return zones, boundaries


def address_matrices(cascade: Cascade, family: str) -> dict[str, np.ndarray]:
                                                                                 
    if family == "path_free":
        return {tier: matrix.copy() for tier, matrix in cascade.prefixes.items()}
    out: dict[str, np.ndarray] = {}
    previous: np.ndarray | None = None
    for tier in TIERS:
        current = cascade.prefixes[tier]
        if previous is None:
            combined = current.copy()
        else:
            width = min(previous.shape[1], current.shape[1])
            combined = np.empty((len(current), width), dtype=object)
            for column in range(width):
                combined[:, column] = np.asarray([
                    f"{left}>{right}" for left, right in zip(previous[:, column], current[:, column])
                ], dtype=object)
        out[tier] = combined
        previous = combined
    return out


def search_family(
        family: str, frame: pd.DataFrame, calibrated: CalibratedCascade,
        zones: dict[str, np.ndarray], boundaries: dict[str, dict[str, float]],
        addresses: dict[str, np.ndarray], terminal_reference: TerminalReference,
        terminal_action: np.ndarray, contract: Contract,
        global_config: dict[str, Any],
        exact_solver: Callable[[Representation], Any | None],
        route_signature: Callable[[Any], str], construction_mask: np.ndarray
        ) -> list[Representation]:
    cfg = global_config["structure"]
    states = {
        tier: np.asarray([f"{tier}|D={zone}|R" for zone in zones[tier]], dtype=object)
        for tier in TIERS
    }
    rule_registry: dict[str, Any] = {
        "decision_boundaries": boundaries,
        "model_address_family": family,
        "balanced_score_proposals": True,
    }
    admissions: list[AdmissionRecord] = []
    base = finalize_representation(
        f"{family}_00", family, copy_states(states), admissions,
        construction_mask, rule_registry)
    current_policy = exact_solver(base)
    if current_policy is None:
        return []
    snapshots = [base]
    used: set[tuple[str, str, str]] = set()
    per_parent: dict[tuple[str, str], int] = {}
    max_total = int(cfg["max_total_admissions_per_class"])
    max_rounds = int(cfg["max_sequential_rounds"])
    max_probes = int(cfg.get("exact_probe_budget_per_step", 24))
    tolerance = float(cfg.get("exact_objective_tolerance", 1e-10))

    for _round in range(max_rounds):
        accepted_in_round = 0
        while len(admissions) < max_total:
            candidates = proposal_candidates(
                family, frame, calibrated, states, addresses, terminal_reference,
                terminal_action, contract, global_config, policy_dual(current_policy),
                construction_mask, admissions, used, per_parent)
            if not candidates:
                break
            accepted = False
            for record, child_mask, residual_mask, identity in candidates[:max_probes]:
                trial_states = copy_states(states)
                trial_states[record.tier][child_mask] = record.child_state
                trial_states[record.tier][residual_mask] = record.residual_state
                trial_admissions = list(admissions) + [record]
                trial = finalize_representation(
                    f"{family}_{len(trial_admissions):02d}", family,
                    trial_states, trial_admissions, construction_mask, rule_registry)
                trial_policy = exact_solver(trial)
                used.add(identity)
                if trial_policy is None:
                    continue
                before = float(current_policy.construction_objective)
                after = float(trial_policy.construction_objective)
                changed = route_signature(trial_policy) != route_signature(current_policy)
                improved = after < before - max(tolerance, tolerance * abs(before))
                if not (changed and improved):
                    continue
                record.exact_objective_before = before
                record.exact_objective_after = after
                record.exact_route_changed = True
                states = trial_states
                admissions = trial_admissions
                current_policy = trial_policy
                per_parent[(record.tier, record.parent_state)] = (
                    per_parent.get((record.tier, record.parent_state), 0) + 1)
                snapshots.append(trial)
                accepted_in_round += 1
                accepted = True
                break
            if not accepted:
                break
        if accepted_in_round == 0:
            break
    return snapshots


def proposal_candidates(
        family: str, frame: pd.DataFrame, calibrated: CalibratedCascade,
        states: dict[str, np.ndarray], addresses: dict[str, np.ndarray],
        terminal_reference: TerminalReference, terminal_action: np.ndarray,
        contract: Contract, global_config: dict[str, Any], dual: dict[str, Any],
        construction_mask: np.ndarray, admissions: list[AdmissionRecord],
        used: set[tuple[str, str, str]], per_parent: dict[tuple[str, str], int]
        ) -> list[tuple[AdmissionRecord, np.ndarray, np.ndarray, tuple[str, str, str]]]:
    cfg = global_config["structure"]
    candidates: list[tuple[AdmissionRecord, np.ndarray, np.ndarray, tuple[str, str, str]]] = []
    for tier in TIERS:
        for parent_state in sorted(np.unique(states[tier][construction_mask])):
            if not str(parent_state).endswith("|R"):
                continue
            parent_key = (tier, str(parent_state))
            if per_parent.get(parent_key, 0) >= int(cfg["max_children_per_parent"]):
                continue
            parent_mask = construction_mask & (states[tier] == parent_state)
            parent_token = sha256(np.flatnonzero(parent_mask).tobytes()).hexdigest()[:12]
            score = calibrated.scores[tier]
            threshold = balanced_threshold(score[parent_mask])
            score_address = f"score<={threshold:.17g}"
            score_identity = (tier, str(parent_state), f"{parent_token}|{score_address}")
            if score_identity not in used and np.isfinite(threshold):
                maybe_add_candidate(
                    candidates, score_identity, "balanced_score", family, tier,
                    str(parent_state), score_address, parent_mask & (score <= threshold),
                    parent_mask & (score > threshold), frame, calibrated,
                    terminal_reference, terminal_action, contract, global_config,
                    dual, len(admissions) + 1)
            for column in range(addresses[tier].shape[1]):
                for address in np.unique(addresses[tier][parent_mask]):
                    leaf_address = f"leaf[{column}]={address}"
                    identity = (tier, str(parent_state), f"{parent_token}|{leaf_address}")
                    if identity in used:
                        continue
                    child = parent_mask & (addresses[tier][:, column] == address)
                    maybe_add_candidate(
                        candidates, identity, "model_address", family, tier,
                        str(parent_state), leaf_address, child, parent_mask & ~child,
                        frame, calibrated, terminal_reference, terminal_action,
                        contract, global_config, dual, len(admissions) + 1)
    candidates.sort(key=lambda item: (
        -item[0].fixed_dual_gain, item[0].tier,
        item[0].parent_state, item[0].address))
    return candidates


def maybe_add_candidate(
        candidates: list, identity: tuple[str, str, str], proposal_type: str,
        family: str, tier: str, parent_state: str, address: str,
        child: np.ndarray, residual: np.ndarray, frame: pd.DataFrame,
        calibrated: CalibratedCascade, terminal_reference: TerminalReference,
        terminal_action: np.ndarray, contract: Contract,
        global_config: dict[str, Any], dual: dict[str, Any],
        admission_index: int) -> None:
    cfg = global_config["structure"]
    if not support_pass(frame, child, residual, cfg):
        return
    record = fixed_dual_ledger(
        family, proposal_type, tier, parent_state, address, child, residual,
        frame, calibrated, terminal_reference, terminal_action, contract,
        global_config, dual, admission_index)
    if record.fixed_dual_gain > float(cfg["min_fixed_dual_gain"]):
        candidates.append((record, child, residual, identity))


def balanced_threshold(values: np.ndarray) -> float:
    clean = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if len(clean) < 2 or clean[0] == clean[-1]:
        return float("nan")
    middle = len(clean) // 2
    left, right = clean[middle - 1], clean[middle]
    if left == right:
        unique = np.unique(clean)
        return float(unique[(len(unique) - 1) // 2]) if len(unique) > 1 else float("nan")
    return float((left + right) / 2.0)


def support_pass(frame: pd.DataFrame, child: np.ndarray, residual: np.ndarray,
                 cfg: dict[str, Any]) -> bool:
    y = frame["target"].to_numpy(int)
    return bool(
        int(child.sum()) >= int(cfg["support_min_n"])
        and int(y[child].sum()) >= int(cfg["support_min_positive_n"])
        and int(residual.sum()) >= int(cfg["residual_min_n"])
        and int(y[residual].sum()) >= int(cfg["residual_min_positive_n"]))


def fixed_dual_ledger(
        family: str, proposal_type: str, tier: str, parent_state: str,
        address: str, child: np.ndarray, residual: np.ndarray,
        frame: pd.DataFrame, calibrated: CalibratedCascade,
        terminal_reference: TerminalReference, terminal_action: np.ndarray,
        contract: Contract, global_config: dict[str, Any],
        anchor_dual: dict[str, Any], admission_index: int) -> AdmissionRecord:
    parent = child | residual
    parent_risk = region_risk(frame, parent, terminal_action)
    child_risk = adjusted_descendant_risk(
        frame, child, residual, terminal_action,
        global_config["structure"]["risk_shrinkage"])
    residual_risk = adjusted_descendant_risk(
        frame, residual, child, terminal_action,
        global_config["structure"]["risk_shrinkage"])
    parent_q = region_action_values(
        tier, parent, parent_risk, frame, calibrated, terminal_reference,
        contract, global_config, anchor_dual)
    child_q = region_action_values(
        tier, child, child_risk, frame, calibrated, terminal_reference,
        contract, global_config, anchor_dual)
    residual_q = region_action_values(
        tier, residual, residual_risk, frame, calibrated, terminal_reference,
        contract, global_config, anchor_dual)
    parent_action = min(parent_q, key=lambda action: parent_q[action]["total"])
    child_action = min(child_q, key=lambda action: child_q[action]["total"])
    residual_action = min(residual_q, key=lambda action: residual_q[action]["total"])
    parent_n = int(parent.sum())
    child_weight = float(child.sum()) / parent_n
    residual_weight = float(residual.sum()) / parent_n
    face_value_per_parent = (
        parent_q[parent_action]["economic"]
        - child_weight * child_q[child_action]["economic"]
        - residual_weight * residual_q[residual_action]["economic"])
    construction_n = max(int(frame["split"].eq("mdp_build").sum()), 1)
    parent_mass = parent_n / construction_n
    face_value = parent_mass * face_value_per_parent
    hinge_mass = cvar_hinge_discrepancy(
        tier, child, residual, frame, calibrated, terminal_action,
        float(anchor_dual.get("tail_eta", global_config["audit"]["report_cap"])),
        float(global_config["audit"]["tail_mass"]))
    charge = float(anchor_dual.get("tail_shadow_price", 0.0)) * parent_mass * hinge_mass
    gain = float(face_value - charge)
    if len({parent_action, child_action, residual_action}) == 1:
        gain = min(gain, 0.0)
    token = sha256(f"{tier}|{parent_state}|{address}".encode("utf-8")).hexdigest()[:10]
    base = parent_state[:-2] if parent_state.endswith("|R") else parent_state
    return AdmissionRecord(
        family=family, proposal_type=proposal_type, tier=tier,
        parent_state=parent_state, address=address,
        child_state=f"{base}|C={admission_index:02d}:{token}",
        residual_state=f"{base}|R", child_n=int(child.sum()),
        residual_n=int(residual.sum()), parent_action=parent_action,
        child_action=child_action, residual_action=residual_action,
        action_face_value=float(face_value), cvar_resolution_charge=float(charge),
        fixed_dual_gain=gain)


def cvar_hinge_discrepancy(
        tier: str, child: np.ndarray, residual: np.ndarray,
        frame: pd.DataFrame, calibrated: CalibratedCascade,
        terminal_action: np.ndarray, eta: float, rho: float) -> float:
    parent = child | residual
    parent_n = max(int(parent.sum()), 1)
    labels = frame["target"].to_numpy(int)
    event = (labels == 1) & (terminal_action != "NEG")
    discrepancy = 0.0
    for public_bin in np.unique(calibrated.public_bins[tier][parent]):
        in_bin = parent & (calibrated.public_bins[tier] == public_bin)
        bin_n = int(in_bin.sum())
        if not bin_n:
            continue
        q_parent = float(np.mean(event[in_bin]))
        mixture = 0.0
        for part in (in_bin & child, in_bin & residual):
            part_n = int(part.sum())
            if part_n:
                mixture += (part_n / bin_n) * max(float(np.mean(event[part])) - eta, 0.0)
        local = max(0.0, mixture - max(q_parent - eta, 0.0))
        discrepancy += (bin_n / parent_n) * local / max(rho, 1e-12)
    return float(discrepancy)


def region_action_values(
        tier: str, mask: np.ndarray, risk: float, frame: pd.DataFrame,
        calibrated: CalibratedCascade, terminal_reference: TerminalReference,
        contract: Contract, global_config: dict[str, Any],
        anchor_dual: dict[str, Any]) -> dict[str, dict[str, float]]:
    p = float(np.mean(frame.loc[mask, "target"].to_numpy(int)))
    base = float(contract.tier_costs[tier])
    eta = float(anchor_dual.get("tail_eta", global_config["audit"]["report_cap"]))
    shadow = float(anchor_dual.get("tail_shadow_price", 0.0))
    rho = float(global_config["audit"]["tail_mass"])
    cap = float(global_config["audit"]["report_cap"])
    tail = eta + max(risk - eta, 0.0) / rho - cap
    values: dict[str, dict[str, float]] = {
        "NEG": {"economic": base, "risk": shadow * tail},
        "POS": {"economic": base + contract.false_positive_price * (1.0 - p), "risk": 0.0},
        "REVIEW": {"economic": base + contract.review_price, "risk": 0.0}}
    if tier != TIERS[-1]:
        next_tier = TIERS[TIERS.index(tier) + 1]
        flow_values = anchor_dual.get("flow_values", {})
        state_arrays = anchor_dual.get("states")
        if state_arrays is None:
            next_states = [f"{next_tier}|B={int(value):02d}"
                           for value in calibrated.public_bins[next_tier][mask]]
        else:
            next_states = np.asarray(state_arrays[next_tier], dtype=object)[mask].astype(str)
        continuation = [float(flow_values.get(f"{next_tier}::{state}", 1.0e9))
                        for state in next_states]
        values["CONTINUE"] = {"economic": float(np.mean(continuation)), "risk": 0.0}
    for value in values.values():
        value["total"] = float(value["economic"] + value["risk"])
    return values


def region_risk(frame: pd.DataFrame, mask: np.ndarray,
                terminal_action: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    event = (frame["target"].to_numpy(int) == 1) & (terminal_action != "NEG")
    return float(np.mean(event[mask]))


def adjusted_descendant_risk(frame: pd.DataFrame, target: np.ndarray,
                             complement: np.ndarray, terminal_action: np.ndarray,
                             shrinkage: dict[str, Any]) -> float:
    raw = region_risk(frame, target, terminal_action)
    if not bool(shrinkage.get("enabled", False)):
        return raw
    parent = region_risk(frame, complement, terminal_action)
    n = max(int(target.sum()), 1)
    difference = abs(raw - parent)
    if str(shrinkage.get("tau_mode", "fixed")) == "adaptive_variance":
        variance = max(parent * (1.0 - parent), 1.0 / n)
        tau = n * variance / max(difference * difference, 1e-12)
        tau = float(np.clip(tau, float(shrinkage["tau_min"]), float(shrinkage["tau_max"])))
    else:
        tau = float(shrinkage["tau"])
    pooled = (raw * n + parent * tau) / (n + tau)
    return float(max(raw, pooled) if str(shrinkage.get("one_sided", "none")) == "up" else pooled)


def policy_dual(policy: Any) -> dict[str, Any]:
    return {
        "tail_eta": float(policy.tail_eta),
        "tail_shadow_price": float(policy.tail_shadow_price),
        "flow_values": dict(policy.flow_values),
        "states": policy.representation.states}


def copy_states(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {tier: values.copy() for tier, values in states.items()}


def finalize_representation(representation_id: str, family: str,
                            states: dict[str, np.ndarray],
                            admissions: list[AdmissionRecord],
                            construction_mask: np.ndarray,
                            rule_registry: dict[str, Any] | None = None) -> Representation:
    rules = dict(rule_registry or {})
    digest = sha256()
    digest.update(family.encode("utf-8"))
    digest.update(content_hash(rules).encode("ascii"))
    for tier in TIERS:
        for value in states[tier][construction_mask].astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    admission_rules = [{
        "proposal_type": row.proposal_type,
        "tier": row.tier,
        "parent_state": row.parent_state,
        "address": row.address,
        "child_state": row.child_state,
        "residual_state": row.residual_state,
    } for row in admissions]
    digest.update(content_hash(admission_rules).encode("ascii"))
    return Representation(
        representation_id, family, states, list(admissions), rules, digest.hexdigest())
