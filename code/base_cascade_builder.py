                                                                                      

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from all_tasks_global_parameters import TIERS, TaskSpec


@dataclass
class TabularEncoder:
    columns: list[str]
    numeric: dict[str, float]
    categorical: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "TabularEncoder":
        numeric: dict[str, float] = {}
        categorical: dict[str, dict[str, int]] = {}
        for column in columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                numeric[column] = float(values.median()) if values.notna().any() else 0.0
            else:
                levels = sorted(series.dropna().astype(str).unique())
                categorical[column] = {value: index for index, value in enumerate(levels)}
        return cls(list(columns), numeric, categorical)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for column in self.columns:
            if column in self.numeric:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(self.numeric[column])
                blocks.append(values.to_numpy(dtype=np.float32, copy=False))
            else:
                mapping = self.categorical[column]
                values = frame[column].astype("string").map(mapping).fillna(-1)
                blocks.append(values.to_numpy(dtype=np.float32, copy=False))
        return np.column_stack(blocks) if blocks else np.empty((len(frame), 0), dtype=np.float32)


@dataclass
class TierModel:
    tier: str
    features: list[str]
    encoder: TabularEncoder
    model: LGBMClassifier
    leaf_prefix_depths: tuple[int, ...]

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.encoder.transform(frame)
        return np.asarray(self.model.booster_.predict(matrix), dtype=float)

    def prefixes(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.encoder.transform(frame)
        leaves = np.asarray(self.model.booster_.predict(matrix, pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        depths = [depth for depth in self.leaf_prefix_depths if depth <= leaves.shape[1]]
        if not depths:
            raise RuntimeError("no configured leaf-prefix depth is supported by the fitted cascade")
        out = np.empty((len(frame), len(depths)), dtype=object)
        for column, depth in enumerate(depths):
            values = pd.DataFrame(np.asarray(leaves[:, :depth], dtype=np.int32))
            hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64)
            out[:, column] = [f"leaf_d{depth}={int(value):016x}" for value in hashes]
        return out


@dataclass
class Cascade:
    models: dict[str, TierModel]
    raw_scores: dict[str, np.ndarray]
    prefixes: dict[str, np.ndarray]

    def current_signatures(self) -> dict[str, np.ndarray]:
        return {tier: signature_rows(matrix) for tier, matrix in self.prefixes.items()}

    def cumulative_signatures(self) -> dict[str, np.ndarray]:
        current = self.current_signatures()
        out: dict[str, np.ndarray] = {}
        history = np.full(len(next(iter(current.values()))), "", dtype=object)
        for tier in TIERS:
            history = np.asarray([
                f"{left}>{right}" if left else str(right)
                for left, right in zip(history, current[tier])
            ], dtype=object)
            out[tier] = history.copy()
        return out


def fit_cascade(frame: pd.DataFrame, spec: TaskSpec,
                global_config: dict[str, Any]) -> Cascade:
    cfg = global_config["cascade"]
    fit = frame["split"].eq("model_train").to_numpy()
    if not np.any(fit):
        raise ValueError("Cascade Training split is empty")
    y = frame.loc[fit, "target"].to_numpy(int)
    models: dict[str, TierModel] = {}
    raw_scores: dict[str, np.ndarray] = {}
    prefixes: dict[str, np.ndarray] = {}
    for tier_index, tier in enumerate(TIERS):
        features = spec.cumulative_features[tier]
        encoder = TabularEncoder.fit(frame.loc[fit], features)
        train_x = encoder.transform(frame.loc[fit])
        model = LGBMClassifier(
            objective="binary",
            learning_rate=float(cfg["learning_rate"]),
            n_estimators=int(cfg["n_estimators"]),
            num_leaves=int(cfg["num_leaves"]),
            min_child_samples=int(cfg["min_child_samples"]),
            random_state=int(global_config["random_seed"]) + tier_index,
            n_jobs=int(cfg.get("n_jobs", 1)),
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(train_x, y)
        tier_model = TierModel(
            tier=tier,
            features=features,
            encoder=encoder,
            model=model,
            leaf_prefix_depths=tuple(int(value) for value in cfg["leaf_prefix_depths"]),
        )
        models[tier] = tier_model
        raw_scores[tier] = tier_model.scores(frame)
        prefixes[tier] = tier_model.prefixes(frame)
    return Cascade(models=models, raw_scores=raw_scores, prefixes=prefixes)


def signature_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("prefix matrix must be two-dimensional")
    return np.asarray([";".join(map(str, row)) for row in matrix], dtype=object)
