"""Transparent Phase 2 ridge-regression baseline and validation diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
# Standard biochemical descriptors, used only as compact mutation descriptors.
HYDROPATHY = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}
SIDE_CHAIN_VOLUME = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6}
CHARGE_PH7 = {"D": -1.0, "E": -1.0, "H": 0.1, "K": 1.0, "R": 1.0}
FEATURE_NAMES = tuple(
    [f"wild_type_{residue}" for residue in AMINO_ACIDS]
    + [f"mutant_{residue}" for residue in AMINO_ACIDS]
    + ["normalized_position", "sequence_length", "delta_hydropathy", "delta_side_chain_volume", "delta_charge_ph7"]
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0.0] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class RidgeModel:
    alpha: float
    intercept: float
    weights: np.ndarray
    standardizer: Standardizer

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(values) @ self.weights + self.intercept


def feature_matrix(rows: Iterable[dict[str, str]]) -> np.ndarray:
    rows = list(rows)
    values = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for index, row in enumerate(rows):
        wild_type = row["wild_type_residue"]
        mutant = row["mutant_residue"]
        if wild_type not in AA_INDEX or mutant not in AA_INDEX:
            raise ValueError(f"Unknown residue in Phase 1 record: {wild_type}->{mutant}")
        sequence_length = len(row["wild_type_sequence"])
        position = int(row["position"])
        if not 1 <= position <= sequence_length:
            raise ValueError(f"Invalid position {position} for sequence length {sequence_length}")
        values[index, AA_INDEX[wild_type]] = 1.0
        values[index, len(AMINO_ACIDS) + AA_INDEX[mutant]] = 1.0
        values[index, 40] = position / sequence_length
        values[index, 41] = float(sequence_length)
        values[index, 42] = HYDROPATHY[mutant] - HYDROPATHY[wild_type]
        values[index, 43] = SIDE_CHAIN_VOLUME[mutant] - SIDE_CHAIN_VOLUME[wild_type]
        values[index, 44] = CHARGE_PH7.get(mutant, 0.0) - CHARGE_PH7.get(wild_type, 0.0)
    return values


def targets(rows: Iterable[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["ddg_kcal_mol"]) for row in rows], dtype=np.float64)


def fit_ridge(values: np.ndarray, labels: np.ndarray, alpha: float) -> RidgeModel:
    if alpha < 0:
        raise ValueError("Ridge alpha must be nonnegative")
    standardizer = Standardizer.fit(values)
    transformed = standardizer.transform(values)
    label_mean = float(labels.mean())
    centered_labels = labels - label_mean
    covariance = transformed.T @ transformed / len(transformed)
    cross_product = transformed.T @ centered_labels / len(transformed)
    weights = np.linalg.solve(covariance + alpha * np.eye(transformed.shape[1]), cross_product)
    return RidgeModel(alpha=alpha, intercept=label_mean, weights=weights, standardizer=standardizer)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = math.sqrt(float(first_centered @ first_centered) * float(second_centered @ second_centered))
    return None if denominator == 0.0 else float(first_centered @ second_centered / denominator)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    residuals = predicted - observed
    return {
        "count": int(len(observed)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "pearson": _pearson(observed, predicted),
        "spearman": _pearson(_average_ranks(observed), _average_ranks(predicted)),
    }
