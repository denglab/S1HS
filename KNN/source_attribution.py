"""Utilities for KNN-based source-attribution subsets.

The manuscript notebooks classify human isolates into food-animal, wild-animal,
and generalist groups before re-running downstream burden analyses. This module
keeps that logic in a reusable place so scripts do not have to copy notebook
cells.
"""

from __future__ import annotations

import ast
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_DIST_CUTOFF = 38
DEFAULT_TRUST_CUTOFF = 0.54


def parse_vector(value) -> list[float]:
    """Parse a stored neighbor-distance or neighbor-label vector."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return [float(x) for x in parsed]
            return [float(parsed)]
        except (SyntaxError, ValueError):
            text = text.strip("[]")
            if not text:
                return []
            return [float(x.strip()) for x in text.split(",") if x.strip()]
    return [float(value)]


def vector_value(value, index: int = 0, default: float = np.nan) -> float:
    """Return one indexed value from a serialized vector."""
    vector = parse_vector(value)
    if index < 0 or index >= len(vector):
        return default
    return vector[index]


def _existing_animal_type_to_class(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", " ")
    if normalized == "food animal":
        return "F"
    if normalized == "wild animal":
        return "W"
    if normalized == "generalist":
        return "G"
    return None


def classify_human_isolate(
    row: pd.Series,
    *,
    prediction_col: str = "Prediction",
    trust_col: str = "Trust (rounded)",
    distance_col: str = "Dist (int)",
    dist_to_prefix: str = "Dist to ",
    animal_type_col: Optional[str] = "animal_type",
    prefer_existing_animal_type: bool = True,
    nn_index_food_animal: int = 0,
    nn_index_wild_animal: int = 0,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    trust_cutoff: float = DEFAULT_TRUST_CUTOFF,
) -> str:
    """Classify one human isolate as food-animal (F), wild-animal (W), or generalist (G).

    When the metadata already contains the notebook-derived ``animal_type``
    column, the default is to reuse it. Otherwise, this follows the notebook
    threshold logic:

    - F: nearest distance to the predicted food-animal class is <= ``dist_cutoff``
      and trust is >= ``trust_cutoff``.
    - W: nearest overall neighbor distance is > ``dist_cutoff``.
    - G: neither F nor W.
    """
    if prefer_existing_animal_type and animal_type_col in row.index:
        existing = _existing_animal_type_to_class(row.get(animal_type_col))
        if existing is not None:
            return existing

    pred = row.get(prediction_col)
    if pd.isna(pred):
        return "G"
    pred = int(float(pred))

    trust = row.get(trust_col, np.nan)
    trust = np.nan if pd.isna(trust) else float(trust)

    predicted_distance_col = f"{dist_to_prefix}{pred}"
    if predicted_distance_col in row.index and pd.notna(row.get(predicted_distance_col)):
        food_distance = vector_value(row.get(predicted_distance_col), nn_index_food_animal)
    else:
        food_distance = vector_value(row.get(distance_col), nn_index_food_animal)

    wild_distance = vector_value(row.get(distance_col), nn_index_wild_animal)

    is_food = pd.notna(food_distance) and food_distance <= dist_cutoff and pd.notna(trust) and trust >= trust_cutoff
    is_wild = pd.notna(wild_distance) and wild_distance > dist_cutoff

    if is_food:
        return "F"
    if is_wild:
        return "W"
    return "G"


def add_human_attribution_class(
    df: pd.DataFrame,
    *,
    node_col: str = "curated_source_region",
    human_label: str = "human",
    output_col: str = "human_attr_class",
    **classification_kwargs,
) -> pd.DataFrame:
    """Return a copy with F/W/G labels added for human rows."""
    out = df.copy()
    out[output_col] = pd.NA
    human_mask = out[node_col] == human_label
    if human_mask.any():
        out.loc[human_mask, output_col] = out.loc[human_mask].apply(
            lambda row: classify_human_isolate(row, **classification_kwargs),
            axis=1,
        )
    return out


def filter_by_human_attribution(
    df: pd.DataFrame,
    selected_human_isolates: Optional[str] = None,
    *,
    node_col: str = "curated_source_region",
    human_label: str = "human",
    output_col: str = "human_attr_class",
    **classification_kwargs,
) -> pd.DataFrame:
    """Keep all non-human rows and a selected subset of human rows.

    ``selected_human_isolates`` can be:

    - ``None`` or ``"all"``: keep all human isolates.
    - ``"F"``: keep food-animal-attributed human isolates.
    - ``"G"``: keep generalist human isolates.
    - ``"FG"``: keep food-animal-attributed and generalist human isolates.
    """
    if selected_human_isolates is None:
        return df.copy()

    selected = str(selected_human_isolates).upper()
    if selected == "ALL":
        return df.copy()
    if selected not in {"F", "G", "FG"}:
        raise ValueError("selected_human_isolates must be one of None, 'all', 'F', 'G', or 'FG'.")

    with_class = add_human_attribution_class(
        df,
        node_col=node_col,
        human_label=human_label,
        output_col=output_col,
        **classification_kwargs,
    )

    human_mask = with_class[node_col] == human_label
    if selected == "FG":
        keep_human = with_class[output_col].isin(["F", "G"])
    else:
        keep_human = with_class[output_col] == selected

    keep = ~human_mask | keep_human
    return with_class.loc[keep].drop(columns=[output_col])


def attribution_counts(
    df: pd.DataFrame,
    *,
    node_col: str = "curated_source_region",
    human_label: str = "human",
    **classification_kwargs,
) -> pd.Series:
    """Count F/W/G labels among human isolates."""
    with_class = add_human_attribution_class(
        df,
        node_col=node_col,
        human_label=human_label,
        **classification_kwargs,
    )
    return with_class.loc[with_class[node_col] == human_label, "human_attr_class"].value_counts(dropna=False)


def normalize_subset_names(values: Iterable[str]) -> list[str]:
    """Normalize CLI/user subset names while preserving order."""
    normalized = []
    for value in values:
        item = str(value).upper()
        if item == "ALL":
            item = "all"
        if item not in {"all", "F", "G", "FG"}:
            raise ValueError(f"Unknown human subset: {value!r}")
        normalized.append(item)
    return normalized
