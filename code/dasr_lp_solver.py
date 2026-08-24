                                                                                 

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from all_tasks_global_parameters import ACTIONS, Contract, TIERS, canonical_json, content_hash
from base_cascade_calibration import CalibratedCascade
from dasr_structure_resolution import Representation


@dataclass
class Policy:
    policy_id: str
    family: str
    representation: Representation
    fractions: dict[str, dict[str, dict[str, float]]]
    construction_objective: float
    tail_eta: float
    tail_shadow_price: float
    flow_residual_max: float
    policy_hash: str
    flow_values: dict[str, float] = field(default_factory=dict)
    frozen_before_selection: bool = True

    def public(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "family": self.family,
            "representation_id": self.representation.representation_id,
            "representation_fingerprint": self.representation.fingerprint,
            "policy_hash": self.policy_hash,
            "construction_objective": self.construction_objective,
            "tail_eta": self.tail_eta,
            "tail_shadow_price": self.tail_shadow_price,
            "flow_residual_max": self.flow_residual_max,
            "flow_values": self.flow_values,
            "frozen_before_selection": self.frozen_before_selection,
            "fractions": self.fractions,
        }


def solve_representation(frame: pd.DataFrame, calibrated: CalibratedCascade,
                         terminal_action: np.ndarray, representation: Representation,
                         contract: Contract, global_config: dict[str, Any]) -> Policy | None:
    construction = frame["split"].eq("mdp_build").to_numpy()
    stats = state_statistics(frame, calibrated, terminal_action, representation,
                             construction, global_config)
    transitions = transition_tables(representation, construction)
    eta_grid = cvar_eta_grid(stats, global_config)
    solutions = [solve_at_eta(stats, transitions, representation, construction,
                              contract, global_config, eta)
                 for eta in eta_grid]
    feasible = [solution for solution in solutions if solution is not None]
    if not feasible:
        return None
    best = min(feasible, key=lambda row: (row["objective"], row["eta"]))
    fractions = fractional_policy(best["rows"], best["values"])
    definition = {
        "family": representation.family,
        "representation_fingerprint": representation.fingerprint,
        "fractions": fractions,
    }
    policy_hash = content_hash(definition)
    return Policy(
        policy_id=f"{representation.representation_id}__{policy_hash[:10]}",
        family=representation.family,
        representation=representation,
        fractions=fractions,
        construction_objective=float(best["objective"]),
        tail_eta=float(best["eta"]),
        tail_shadow_price=float(best["shadow_price"]),
        flow_residual_max=float(best["flow_residual_max"]),
        policy_hash=policy_hash,
        flow_values=dict(best["flow_values"]),
    )


def fail_safe_policy(representation: Representation) -> Policy:
    fractions = {
        tier: {str(state): {"REVIEW": 1.0} for state in np.unique(representation.states[tier])}
        for tier in TIERS
    }
    definition = {
        "family": "fail_safe",
        "representation_fingerprint": representation.fingerprint,
        "fractions": fractions,
    }
    policy_hash = content_hash(definition)
    return Policy(
        policy_id=f"fail_safe__{policy_hash[:10]}",
        family="fail_safe",
        representation=representation,
        fractions=fractions,
        construction_objective=1.0e30,
        tail_eta=0.0,
        tail_shadow_price=0.0,
        flow_residual_max=0.0,
        policy_hash=policy_hash,
    )


def state_statistics(frame: pd.DataFrame, calibrated: CalibratedCascade,
                     terminal_action: np.ndarray, representation: Representation,
                     construction: np.ndarray,
                     global_config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    labels = frame["target"].to_numpy(int)
    excess = (labels == 1) & (terminal_action != "NEG")
    shrinkage = global_config["structure"]["risk_shrinkage"]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for tier in TIERS:
        tier_rows: dict[str, dict[str, Any]] = {}
        for state in sorted(np.unique(representation.states[tier][construction])):
            mask = construction & (representation.states[tier] == state)
            state_n = int(mask.sum())
            terms: dict[int, dict[str, float]] = {}
            for public_bin in range(int(global_config["audit"]["score_bins_per_tier"])):
                slice_mask = mask & (calibrated.public_bins[tier] == public_bin)
                slice_n = int(slice_mask.sum())
                if slice_n == 0:
                    continue
                raw_risk = float(np.mean(excess[slice_mask]))
                risk = adjusted_state_slice_risk(
                    raw_risk, state, tier, public_bin, mask, slice_mask, construction,
                    representation, calibrated, excess, shrinkage)
                terms[public_bin] = {
                    "frac": slice_n / state_n,
                    "risk": risk,
                    "raw_risk": raw_risk,
                    "n": slice_n,
                }
            tier_rows[str(state)] = {
                "n": state_n,
                "mass": state_n / max(int(construction.sum()), 1),
                "p_neg": float(np.mean(labels[mask] == 0)),
                "excess_slice_terms": terms,
            }
        out[tier] = tier_rows
    return out


def adjusted_state_slice_risk(raw: float, state: str, tier: str, public_bin: int,
                              state_mask: np.ndarray, slice_mask: np.ndarray,
                              construction: np.ndarray, representation: Representation,
                              calibrated: CalibratedCascade, excess: np.ndarray,
                              shrinkage: dict[str, Any]) -> float:
    if not bool(shrinkage.get("enabled", False)) or "|C=" not in str(state):
        return raw
    zone = str(state).split("|C=", 1)[0]
    zone_mask = construction & np.asarray([
        str(value).startswith(zone) for value in representation.states[tier]
    ], dtype=bool)
    complement = zone_mask & ~state_mask & (calibrated.public_bins[tier] == public_bin)
    if not np.any(complement):
        return raw
    parent = float(np.mean(excess[complement]))
    n = max(int(slice_mask.sum()), 1)
    if str(shrinkage.get("tau_mode", "fixed")) == "adaptive_variance":
        variance = max(parent * (1.0 - parent), 1.0 / n)
        tau = n * variance / max((raw - parent) ** 2, 1e-12)
        tau = float(np.clip(tau, float(shrinkage["tau_min"]), float(shrinkage["tau_max"])))
    else:
        tau = float(shrinkage["tau"])
    pooled = (n * raw + tau * parent) / (n + tau)
    return float(max(raw, pooled) if shrinkage.get("one_sided") == "up" else pooled)


def transition_tables(representation: Representation,
                      construction: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tier in TIERS[:-1]:
        next_tier = TIERS[TIERS.index(tier) + 1]
        table: dict[str, dict[str, float]] = {}
        for state in np.unique(representation.states[tier][construction]):
            mask = construction & (representation.states[tier] == state)
            next_values, counts = np.unique(representation.states[next_tier][mask], return_counts=True)
            table[str(state)] = {str(value): int(count) / int(mask.sum())
                                 for value, count in zip(next_values, counts)}
        out[tier] = table
    return out


def cvar_eta_grid(stats: dict[str, dict[str, dict[str, Any]]],
                  global_config: dict[str, Any]) -> np.ndarray:
    risks = [float(term["risk"]) for tier in stats.values() for row in tier.values()
             for term in row["excess_slice_terms"].values()]
    size = int(global_config["solver"]["cvar_eta_grid_size"])
    base = np.asarray(risks or [float(global_config["audit"]["report_cap"])], dtype=float)
    grid = np.quantile(base, np.linspace(0.0, 1.0, size))
    return np.unique(np.r_[0.0, grid, float(global_config["audit"]["report_cap"])])


def solve_at_eta(stats: dict[str, dict[str, dict[str, Any]]],
                 transitions: dict[str, dict[str, dict[str, float]]],
                 representation: Representation, construction: np.ndarray,
                 contract: Contract, global_config: dict[str, Any],
                 eta: float) -> dict[str, Any] | None:
    rows = variable_rows(stats, contract, global_config, eta)
    index = {(row["tier"], row["state"], row["action"]): position
             for position, row in enumerate(rows)}
    keys = [(tier, state) for tier in TIERS for state in sorted(stats[tier])]
    a_eq = np.zeros((len(keys), len(rows)), dtype=float)
    for row_index, (tier, state) in enumerate(keys):
        for action in allowed_actions(tier):
            a_eq[row_index, index[(tier, state, action)]] = 1.0
        if tier != TIERS[0]:
            previous = TIERS[TIERS.index(tier) - 1]
            for previous_state, probabilities in transitions.get(previous, {}).items():
                probability = float(probabilities.get(state, 0.0))
                if probability:
                    a_eq[row_index, index[(previous, previous_state, "CONTINUE")]] -= probability
    initial = initial_distribution(representation.states[TIERS[0]][construction])
    b_eq = np.asarray([initial.get(state, 0.0) if tier == TIERS[0] else 0.0
                       for tier, state in keys], dtype=float)
    a_ub = np.asarray([[float(row["tail_load"]) for row in rows]], dtype=float)
    b_ub = np.asarray([0.0], dtype=float)
    objective = np.asarray([float(row["cost"]) for row in rows], dtype=float)
    result = linprog(
        objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=(0.0, None), method=str(global_config["solver"]["method"]),
    )
    if not result.success:
        return None
    values = np.asarray(result.x, dtype=float)
    flow_residual = a_eq @ values - b_eq
    shadow = float(max(0.0, -np.asarray(result.ineqlin.marginals, dtype=float)[0]))
    equality = np.asarray(result.eqlin.marginals, dtype=float)
    return {
        "rows": rows,
        "values": values,
        "eta": float(eta),
        "objective": float(result.fun),
        "shadow_price": shadow,
        "tail_slack": float(b_ub[0] - (a_ub @ values)[0]),
        "flow_residual_max": float(np.max(np.abs(flow_residual))) if len(flow_residual) else 0.0,
        "flow_values": {f"{tier}::{state}": float(equality[index])
                        for index, (tier, state) in enumerate(keys)},
    }


def variable_rows(stats: dict[str, dict[str, dict[str, Any]]],
                  contract: Contract, global_config: dict[str, Any],
                  eta: float) -> list[dict[str, Any]]:
    rows = []
    for tier in TIERS:
        for state, state_row in stats[tier].items():
            for action in allowed_actions(tier):
                rows.append({
                    "tier": tier,
                    "state": state,
                    "action": action,
                    "cost": action_cost(tier, action, state_row, contract),
                    "tail_load": tail_load(action, state_row, eta, global_config),
                })
    return rows


def action_cost(tier: str, action: str, state_row: dict[str, Any],
                contract: Contract) -> float:
    if action == "CONTINUE":
        return 0.0
    base = float(contract.tier_costs[tier])
    if action == "POS":
        return base + contract.false_positive_price * float(state_row["p_neg"])
    if action == "REVIEW":
        return base + contract.review_price
    return base


def tail_load(action: str, state_row: dict[str, Any], eta: float,
              global_config: dict[str, Any]) -> float:
    if action != "NEG":
        return 0.0
    rho = float(global_config["audit"]["tail_mass"])
    cap = float(global_config["audit"]["report_cap"])
    return float(sum(
        float(term["frac"]) * (eta + max(float(term["risk"]) - eta, 0.0) / rho - cap)
        for term in state_row["excess_slice_terms"].values()
    ))


def allowed_actions(tier: str) -> tuple[str, ...]:
    return ("NEG", "POS", "REVIEW") if tier == TIERS[-1] else ACTIONS


def initial_distribution(states: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(states.astype(str), return_counts=True)
    total = max(int(counts.sum()), 1)
    return {str(value): int(count) / total for value, count in zip(values, counts)}


def fractional_policy(rows: list[dict[str, Any]], values: np.ndarray
                      ) -> dict[str, dict[str, dict[str, float]]]:
    frame = pd.DataFrame(rows).assign(value=np.asarray(values, dtype=float))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tier, tier_rows in frame.groupby("tier", sort=False):
        out[str(tier)] = {}
        for state, state_rows in tier_rows.groupby("state", sort=False):
            total = float(state_rows["value"].sum())
            if total <= 1e-12:
                out[str(tier)][str(state)] = {"REVIEW": 1.0}
            else:
                out[str(tier)][str(state)] = {
                    str(row.action): float(row.value) / total
                    for row in state_rows.itertuples(index=False) if float(row.value) > 1e-12
                }
    return out


def route_policy(frame: pd.DataFrame, calibrated: CalibratedCascade,
                 policy: Policy, mask: np.ndarray | None = None) -> pd.DataFrame:
    selected = np.ones(len(frame), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    source = np.flatnonzero(selected)
    active = np.ones(len(source), dtype=bool)
    stop_tier = np.full(len(source), TIERS[-1], dtype=object)
    stop_state = np.full(len(source), "", dtype=object)
    stop_action = np.full(len(source), "REVIEW", dtype=object)
    reached = {tier: np.zeros(len(source), dtype=bool) for tier in TIERS}
    reached_state = {tier: np.full(len(source), "", dtype=object) for tier in TIERS}
    for tier in TIERS:
        positions = np.flatnonzero(active)
        if len(positions) == 0:
            break
        reached[tier][positions] = True
        global_rows = source[positions]
        states = policy.representation.states[tier][global_rows].astype(str)
        reached_state[tier][positions] = states
        scores = calibrated.scores[tier][global_rows]
        actions = np.full(len(positions), "REVIEW", dtype=object)
        for state in np.unique(states):
            local = np.flatnonzero(states == state)
            shares = policy.fractions.get(tier, {}).get(str(state), {"REVIEW": 1.0})
            actions[local] = stratified_actions(scores[local], shares, final=(tier == TIERS[-1]))
        stop = actions != "CONTINUE"
        stopping_positions = positions[stop]
        stop_tier[stopping_positions] = tier
        stop_state[stopping_positions] = states[stop]
        stop_action[stopping_positions] = actions[stop]
        active[stopping_positions] = False
    return pd.DataFrame({
        "_row_position": source,
        "row_id": frame.iloc[source]["row_id"].to_numpy(object),
        "split": frame.iloc[source]["split"].to_numpy(object),
        "target": frame.iloc[source]["target"].to_numpy(int),
        "tier": stop_tier,
        "state": stop_state,
        "action": stop_action,
        **{f"reached_{tier}": reached[tier] for tier in TIERS},
        **{f"reached_state_{tier}": reached_state[tier] for tier in TIERS},
    })


def stratified_actions(scores: np.ndarray, shares: dict[str, float],
                       final: bool) -> np.ndarray:
    normalized = normalize_shares(shares)
    if final and "CONTINUE" in normalized:
        normalized["REVIEW"] = normalized.get("REVIEW", 0.0) + normalized.pop("CONTINUE")
    counts = share_counts(len(scores), normalized)
    actions = np.full(len(scores), "REVIEW", dtype=object)
    remaining = np.arange(len(scores))
    remaining = assign_extreme(actions, remaining, scores, "NEG", counts.get("NEG", 0), True)
    remaining = assign_extreme(actions, remaining, scores, "POS", counts.get("POS", 0), False)
    remaining = assign_middle(actions, remaining, scores, "CONTINUE", counts.get("CONTINUE", 0))
    return actions


def normalize_shares(shares: dict[str, float]) -> dict[str, float]:
    clean = {str(key): max(float(value), 0.0) for key, value in shares.items()}
    total = sum(clean.values())
    return {"REVIEW": 1.0} if total <= 1e-12 else {key: value / total for key, value in clean.items()}


def share_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {key: value * total for key, value in shares.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    missing = total - sum(counts.values())
    for key in sorted(raw, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:missing]:
        counts[key] += 1
    return counts


def assign_extreme(actions: np.ndarray, remaining: np.ndarray, scores: np.ndarray,
                   action: str, count: int, ascending: bool) -> np.ndarray:
    if count <= 0 or len(remaining) == 0:
        return remaining
    order = np.argsort(scores[remaining], kind="mergesort")
    chosen = remaining[order[:count] if ascending else order[-count:]]
    actions[chosen] = action
    return remaining[~np.isin(remaining, chosen)]


def assign_middle(actions: np.ndarray, remaining: np.ndarray, scores: np.ndarray,
                  action: str, count: int) -> np.ndarray:
    if count <= 0 or len(remaining) == 0:
        return remaining
    center = float(np.median(scores[remaining]))
    order = np.argsort(np.abs(scores[remaining] - center), kind="mergesort")
    chosen = remaining[order[:count]]
    actions[chosen] = action
    return remaining[~np.isin(remaining, chosen)]
