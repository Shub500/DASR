                                                                  

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

from all_tasks_global_parameters import Contract, TIERS
from base_cascade_calibration import CalibratedCascade
from dasr_lp_solver import Policy, route_policy
from dasr_policy_index import IndexedPolicy


@dataclass
class PolicyEvaluation:
    policy_hash: str
    policy_id: str
    family: str
    split: str
    n: int
    cost: float
    consequence_vector: tuple[float, float, float, float, float]
    repriced_cost: float
    tail_risk_observed: float
    tail_risk_ucb: float
    checked_cap: float
    safety_pass: bool
    review_rate: float
    negative_rate: float
    positive_rate: float
    routes: pd.DataFrame

    def public(self) -> dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "policy_id": self.policy_id,
            "family": self.family,
            "split": self.split,
            "n": self.n,
            "cost": self.cost,
            "consequence_vector": list(self.consequence_vector),
            "repriced_cost": self.repriced_cost,
            "tail_risk_observed": self.tail_risk_observed,
            "tail_risk_ucb": self.tail_risk_ucb,
            "checked_cap": self.checked_cap,
            "safety_pass": self.safety_pass,
            "review_rate": self.review_rate,
            "negative_rate": self.negative_rate,
            "positive_rate": self.positive_rate,
        }


def evaluate_policy(frame: pd.DataFrame, calibrated: CalibratedCascade,
                    terminal_action: np.ndarray, policy: Policy,
                    contract: Contract, global_config: dict[str, Any],
                    split: str, cap: float) -> PolicyEvaluation:
    mask = frame["split"].eq(split).to_numpy()
    routes = route_policy(frame, calibrated, policy, mask)
    positions = routes["_row_position"].to_numpy(int)
    labels = frame.iloc[positions]["target"].to_numpy(int)
    actions = routes["action"].to_numpy(object)
    tiers = routes["tier"].to_numpy(object)
    consequence = consequence_vector(labels, actions, routes)
    repriced = float(np.dot(np.asarray(contract.price_vector), np.asarray(consequence)))
    direct = direct_cost(labels, actions, tiers, contract)
    if abs(repriced - direct) > 1e-10:
        raise RuntimeError(f"consequence repricing mismatch: {repriced} vs {direct}")
    observed, ucb = public_tail_audit(
        labels, actions, tiers, positions, terminal_action, calibrated, global_config)
    return PolicyEvaluation(
        policy_hash=policy.policy_hash,
        policy_id=policy.policy_id,
        family=policy.family,
        split=split,
        n=len(routes),
        cost=direct,
        consequence_vector=consequence,
        repriced_cost=repriced,
        tail_risk_observed=observed,
        tail_risk_ucb=ucb,
        checked_cap=float(cap),
        safety_pass=bool(ucb <= float(cap) + 1e-12),
        review_rate=float(np.mean(actions == "REVIEW")) if len(actions) else 0.0,
        negative_rate=float(np.mean(actions == "NEG")) if len(actions) else 0.0,
        positive_rate=float(np.mean(actions == "POS")) if len(actions) else 0.0,
        routes=routes,
    )


def select_policy(frame: pd.DataFrame, calibrated: CalibratedCascade,
                  terminal_action: np.ndarray, entries: list[IndexedPolicy],
                  contract: Contract, global_config: dict[str, Any]
                  ) -> tuple[Policy, PolicyEvaluation, list[PolicyEvaluation]]:
    cap = float(global_config["audit"]["selection_cap"])
    evaluations = [evaluate_policy(
        frame, calibrated, terminal_action, entry.policy, contract,
        global_config, split="post_cal", cap=cap)
        for entry in entries]
    survivors = [row for row in evaluations if row.safety_pass]
    if not survivors:
        raise RuntimeError("frozen frontier has no policy passing the pointwise selection screen")
    selected = min(survivors, key=lambda row: (
        row.cost, row.tail_risk_ucb, row.review_rate, row.policy_hash))
    policy = next(entry.policy for entry in entries if entry.policy.policy_hash == selected.policy_hash)
    return policy, selected, evaluations


def consequence_vector(labels: np.ndarray, actions: np.ndarray,
                       routes: pd.DataFrame) -> tuple[float, float, float, float, float]:
    return (
        float(np.mean(routes["reached_T1"])),
        float(np.mean(routes["reached_T2"])),
        float(np.mean(routes["reached_T3"])),
        float(np.mean((labels == 0) & (actions == "POS"))),
        float(np.mean(actions == "REVIEW")),
    )


def direct_cost(labels: np.ndarray, actions: np.ndarray, tiers: np.ndarray,
                contract: Contract) -> float:
    acquisition = np.asarray([contract.tier_costs[str(tier)] for tier in tiers], dtype=float)
    false_positive = contract.false_positive_price * ((labels == 0) & (actions == "POS"))
    review = contract.review_price * (actions == "REVIEW")
    return float(np.mean(acquisition + false_positive + review)) if len(labels) else 0.0


def public_tail_audit(labels: np.ndarray, actions: np.ndarray, tiers: np.ndarray,
                      positions: np.ndarray, terminal_action: np.ndarray,
                      calibrated: CalibratedCascade,
                      global_config: dict[str, Any]) -> tuple[float, float]:
    alpha = float(global_config["audit"]["total_slice_error"]) / int(
        global_config["audit"]["slice_count"])
    bin_count = int(global_config["audit"]["score_bins_per_tier"])
    rows = []
    total_negative = int(np.sum(actions == "NEG"))
    if total_negative == 0:
        return 0.0, 0.0
    terminal = terminal_action[positions]
    event = (labels == 1) & (actions == "NEG") & (terminal != "NEG")
    for tier in TIERS:
        tier_bins = calibrated.public_bins[tier][positions]
        for public_bin in range(bin_count):
            negative = (actions == "NEG") & (tiers == tier) & (tier_bins == public_bin)
            denominator = int(negative.sum())
            if denominator == 0:
                continue
            events = int(np.sum(event & negative))
            rows.append({
                "weight": denominator / total_negative,
                "observed": events / denominator,
                "ucb": clopper_pearson_upper(events, denominator, alpha),
            })
    rho = float(global_config["audit"]["tail_mass"])
    return weighted_upper_tail(rows, "observed", rho), weighted_upper_tail(rows, "ucb", rho)


def weighted_upper_tail(rows: list[dict[str, float]], field: str, rho: float) -> float:
    remaining = float(rho)
    total = 0.0
    for row in sorted(rows, key=lambda item: (-float(item[field]), -float(item["weight"]))):
        take = min(remaining, float(row["weight"]))
        total += take * float(row[field])
        remaining -= take
        if remaining <= 1e-15:
            break
    return float(total / rho) if rho > 0 else 0.0


def clopper_pearson_upper(events: int, denominator: int, alpha: float) -> float:
    if denominator <= 0:
        return 1.0
    if events >= denominator:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, denominator - events))

