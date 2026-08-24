                                                                                   

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression

from all_tasks_global_parameters import Contract, TIERS
from base_cascade_builder import Cascade


@dataclass
class CalibratedCascade:
    cascade: Cascade
    calibrators: dict[str, IsotonicRegression]
    scores: dict[str, np.ndarray]
    public_bin_edges: dict[str, np.ndarray]
    public_bins: dict[str, np.ndarray]


@dataclass(frozen=True)
class TerminalReference:
    negative_threshold: float
    positive_threshold: float
    false_negative_cap: float
    construction_cap: float
    construction_positive_n: int
    construction_false_negative_n: int


def calibrate_cascade(frame: pd.DataFrame, cascade: Cascade,
                      global_config: dict[str, Any]) -> CalibratedCascade:
    fit = frame["split"].eq("mdp_build").to_numpy()
    y = frame.loc[fit, "target"].to_numpy(int)
    if not np.any(fit):
        raise ValueError("Policy Construction split is empty")
    bins = int(global_config["calibration"]["public_score_bins"])
    calibrators: dict[str, IsotonicRegression] = {}
    scores: dict[str, np.ndarray] = {}
    edges: dict[str, np.ndarray] = {}
    public_bins: dict[str, np.ndarray] = {}
    for tier in TIERS:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(cascade.raw_scores[tier][fit], y)
        calibrated = np.asarray(calibrator.predict(cascade.raw_scores[tier]), dtype=float)
        tier_edges = quantile_edges(calibrated[fit], bins)
        calibrators[tier] = calibrator
        scores[tier] = calibrated
        edges[tier] = tier_edges
        public_bins[tier] = np.searchsorted(tier_edges, calibrated, side="right").astype(int)
    return CalibratedCascade(cascade, calibrators, scores, edges, public_bins)


def fit_terminal_reference(frame: pd.DataFrame, calibrated: CalibratedCascade,
                           contract: Contract,
                           global_config: dict[str, Any]) -> TerminalReference:
    cfg = global_config["terminal_reference"]
    fit = frame["split"].eq(str(cfg["fit_split"])).to_numpy()
    y = frame.loc[fit, "target"].to_numpy(int)
    score = calibrated.scores["T3"][fit]
    public_cap = float(cfg["false_negative_cap"])
    construction_cap = public_cap * float(cfg["construction_reserve"])
    positive_n = int((y == 1).sum())
    allowed = allowed_events(positive_n, construction_cap, float(global_config["audit"]["total_slice_error"]))
    threshold, event_n = largest_negative_threshold(score, y, allowed)
    return TerminalReference(
        negative_threshold=threshold,
        positive_threshold=contract.positive_threshold,
        false_negative_cap=public_cap,
        construction_cap=construction_cap,
        construction_positive_n=positive_n,
        construction_false_negative_n=event_n,
    )


def terminal_actions(calibrated: CalibratedCascade,
                     reference: TerminalReference) -> np.ndarray:
    score = calibrated.scores["T3"]
    actions = np.full(len(score), "REVIEW", dtype=object)
    actions[score <= reference.negative_threshold] = "NEG"
    actions[score >= reference.positive_threshold] = "POS"
    return actions


def quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    raw = np.quantile(clean, np.arange(1, bin_count) / bin_count, method="linear")
    return np.maximum.accumulate(np.asarray(raw, dtype=float))


def largest_negative_threshold(scores: np.ndarray, labels: np.ndarray,
                               allowed_events_n: int) -> tuple[float, int]:
    if allowed_events_n < 0 or len(scores) == 0:
        return -1.0, 0
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ordered_scores = np.asarray(scores, dtype=float)[order]
    positive_cumulative = np.cumsum(np.asarray(labels, dtype=int)[order] == 1)
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(ordered_scores) - 1]
    valid = ends[positive_cumulative[ends] <= allowed_events_n]
    if len(valid) == 0:
        return -1.0, 0
    index = int(valid[-1])
    return float(ordered_scores[index]), int(positive_cumulative[index])


def allowed_events(denominator: int, cap: float, alpha: float) -> int:
    if denominator <= 0:
        return -1
    low, high = -1, denominator
    while low < high:
        middle = (low + high + 1) // 2
        if clopper_pearson_upper(middle, denominator, alpha) <= cap:
            low = middle
        else:
            high = middle - 1
    return low


def clopper_pearson_upper(events: int, denominator: int, alpha: float) -> float:
    if denominator <= 0:
        return 1.0
    if events >= denominator:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, denominator - events))

