"""Animal-feed association analyses for HC clusters.

This script promotes the animal-feed notebook analyses into reusable code. It
tests whether feed-represented HC clusters have broader target-node occupancy,
then optionally evaluates whether feed-linked broad animal clusters are enriched
for human burden.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact, pearsonr
from statsmodels.stats.contingency_tables import StratifiedTable

CURRENT_PATH = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_PATH.parent
DATA_PATH = SRC_ROOT / "data"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from KNN.source_attribution import (  # noqa: E402
    DEFAULT_DIST_CUTOFF,
    DEFAULT_TRUST_CUTOFF,
    filter_by_human_attribution,
    normalize_subset_names,
)


DEFAULT_TARGET_NODES = ["poultry", "swine", "bovine", "companion_animal"]
PET_FOOD_TERMS = (
    "pet",
    "companion",
    "treat",
    "cat",
    "dog",
    "kibble",
    "mouse",
    "pizzle",
    "pig ear",
    "mice",
    "ears",
    "kitten",
    "strips",
    "dragon",
    "canine",
)


@dataclass(frozen=True)
class FeedBreadthSettings:
    cluster_col: str = "HC50"
    node_col: str = "curated_source_region"
    feed_node: str = "animal_feed"
    target_nodes: tuple[str, ...] = tuple(DEFAULT_TARGET_NODES)
    n_perm: int = 2000
    random_seed: int = 186


@dataclass(frozen=True)
class HumanBurdenSettings:
    cluster_col: str = "HC50"
    node_col: str = "curated_source_region"
    feed_node: str = "animal_feed"
    human_node: str = "human"
    animal_target_nodes: tuple[str, ...] = tuple(DEFAULT_TARGET_NODES)
    min_animal_breadth: int = 3
    min_animal_target_total_count: int = 0
    top_fraction: float = 0.10
    human_metric: str = "human_size_adjusted_resid"
    residual_formula: str = "human_count ~ log_total_count + animal_target_breadth"
    n_size_bins: int = 5
    n_perm: int = 10000
    random_seed: int = 42


def read_metadata(path: os.PathLike[str] | str) -> pd.DataFrame:
    """Read a metadata table and normalize HC columns for grouping."""
    df = pd.read_csv(path, low_memory=False)
    for col in ["HC5", "HC10", "HC20", "HC50"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{context} requires missing column(s): {missing_text}")


def add_animal_feed_subclass(
    df: pd.DataFrame,
    *,
    node_col: str = "curated_source_region",
    source_details_col: str = "Source Details",
    output_col: str = "curated_source_region_animal_feed_two_classes",
    feed_node: str = "animal_feed",
    pet_terms: Sequence[str] = PET_FOOD_TERMS,
) -> pd.DataFrame:
    """Split animal feed into pet-food and non-pet-food rows."""
    require_columns(df, [node_col, source_details_col], "Animal-feed subclassing")
    out = df.copy()
    pattern = "|".join(re.escape(term) for term in pet_terms)
    is_feed = out[node_col] == feed_node
    is_pet_food = out[source_details_col].astype("string").str.contains(pattern, case=False, na=False, regex=True)
    out[output_col] = out[node_col]
    out.loc[is_feed & is_pet_food, output_col] = "animal_feed_pet_food"
    out.loc[is_feed & ~is_pet_food, output_col] = "animal_feed_non_pet_food"
    return out


def build_cluster_count_table(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    node_col: str,
    feed_node: str,
    target_nodes: Sequence[str],
    extra_nodes: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the isolate-level working table and cluster-by-node count table."""
    needed = [cluster_col, node_col]
    require_columns(df, needed, "Cluster count table")

    dat = df[needed].dropna(subset=needed).copy()
    dat[cluster_col] = dat[cluster_col].astype(str)
    dat[node_col] = dat[node_col].astype(str)

    count_table = pd.crosstab(dat[cluster_col], dat[node_col])
    required_nodes = [feed_node, *target_nodes, *(extra_nodes or [])]
    for node in required_nodes:
        if node not in count_table.columns:
            count_table[node] = 0
    return dat, count_table


def summarize_feed_breadth(
    count_table: pd.DataFrame,
    *,
    feed_node: str,
    target_nodes: Sequence[str],
) -> pd.DataFrame:
    """Summarize feed representation and target-node breadth for each cluster."""
    summary = pd.DataFrame(index=count_table.index)
    summary["feed_count"] = count_table[feed_node]
    summary["feed_present"] = (summary["feed_count"] > 0).astype(int)
    summary["target_breadth"] = (count_table[list(target_nodes)] > 0).sum(axis=1)
    summary["target_total_count"] = count_table[list(target_nodes)].sum(axis=1)
    summary["total_count"] = count_table.sum(axis=1)
    summary["global_breadth"] = (count_table > 0).sum(axis=1)
    summary["log_total_count"] = np.log1p(summary["total_count"])
    summary["log_feed_count"] = np.log1p(summary["feed_count"])
    return summary


def compute_feed_breadth_statistics(summary: pd.DataFrame, n_targets: int) -> dict[str, float]:
    """Compute observed or permuted feed-breadth statistics."""
    stats: dict[str, float] = {}
    pos = summary.loc[summary["feed_present"] == 1, "target_breadth"]
    neg = summary.loc[summary["feed_present"] == 0, "target_breadth"]

    stats["mean_breadth_diff"] = pos.mean() - neg.mean() if len(pos) and len(neg) else np.nan

    x = summary["log_feed_count"].to_numpy()
    y = summary["target_breadth"].to_numpy()
    stats["feed_count_vs_breadth_r"] = pearsonr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else np.nan

    for k in (1, 2, 3):
        if n_targets >= k:
            stats[f"feed_and_at_least_{k}_targets"] = int(
                ((summary["feed_present"] == 1) & (summary["target_breadth"] >= k)).sum()
            )

    stats["mean_breadth_feed_positive"] = pos.mean() if len(pos) else np.nan
    stats["mean_prop_targets_occupied_feed_positive"] = pos.mean() / n_targets if len(pos) and n_targets else np.nan
    return stats


def empirical_p_greater(observed: float, permuted: Sequence[float]) -> float:
    values = np.asarray(permuted, dtype=float)
    values = values[~np.isnan(values)]
    if np.isnan(observed) or len(values) == 0:
        return np.nan
    return (np.sum(values >= observed) + 1) / (len(values) + 1)


def empirical_p_two_sided(observed: float, permuted: Sequence[float]) -> float:
    values = np.asarray(permuted, dtype=float)
    values = values[~np.isnan(values)]
    if np.isnan(observed) or len(values) == 0:
        return np.nan
    center = values.mean()
    return (np.sum(np.abs(values - center) >= abs(observed - center)) + 1) / (len(values) + 1)


def run_feed_breadth_test(df: pd.DataFrame, settings: FeedBreadthSettings = FeedBreadthSettings()) -> dict[str, object]:
    """Permutation and adjusted-GLM test for feed-associated target-node breadth."""
    target_nodes = list(settings.target_nodes)
    rng = np.random.default_rng(settings.random_seed)

    dat, count_table = build_cluster_count_table(
        df,
        cluster_col=settings.cluster_col,
        node_col=settings.node_col,
        feed_node=settings.feed_node,
        target_nodes=target_nodes,
    )
    cluster_summary = summarize_feed_breadth(
        count_table,
        feed_node=settings.feed_node,
        target_nodes=target_nodes,
    )
    observed_stats = compute_feed_breadth_statistics(cluster_summary, len(target_nodes))

    cluster_values = dat[settings.cluster_col].to_numpy(copy=True)
    node_values = dat[settings.node_col].to_numpy(copy=True)
    perm_rows = []
    for _ in range(settings.n_perm):
        perm_df = pd.DataFrame(
            {
                settings.cluster_col: cluster_values,
                settings.node_col: rng.permutation(node_values),
            }
        )
        perm_count_table = pd.crosstab(perm_df[settings.cluster_col], perm_df[settings.node_col])
        for node in [settings.feed_node, *target_nodes]:
            if node not in perm_count_table.columns:
                perm_count_table[node] = 0
        perm_count_table = perm_count_table.reindex(index=count_table.index, fill_value=0)
        perm_summary = summarize_feed_breadth(
            perm_count_table,
            feed_node=settings.feed_node,
            target_nodes=target_nodes,
        )
        perm_rows.append(compute_feed_breadth_statistics(perm_summary, len(target_nodes)))

    perm_results = pd.DataFrame(perm_rows)
    pval_table = pd.DataFrame(
        {
            stat: {
                "observed": observed_stats[stat],
                "perm_mean": perm_results[stat].mean(skipna=True),
                "perm_sd": perm_results[stat].std(skipna=True),
                "p_one_sided_greater": empirical_p_greater(observed_stats[stat], perm_results[stat]),
                "p_two_sided": empirical_p_two_sided(observed_stats[stat], perm_results[stat]),
            }
            for stat in perm_results.columns
        }
    ).T

    regression_results = {}
    reg_df = cluster_summary.reset_index().rename(columns={settings.cluster_col: "cluster_id"})
    reg_df["prop_target_breadth"] = reg_df["target_breadth"] / len(target_nodes)
    if len(target_nodes) >= 1 and reg_df["target_breadth"].nunique() > 1:
        regression_results["presence_model"] = smf.glm(
            "prop_target_breadth ~ feed_present + log_total_count",
            data=reg_df,
            family=sm.families.Binomial(),
            freq_weights=np.repeat(len(target_nodes), len(reg_df)),
        ).fit()
        regression_results["abundance_model"] = smf.glm(
            "prop_target_breadth ~ log_feed_count + log_total_count",
            data=reg_df,
            family=sm.families.Binomial(),
            freq_weights=np.repeat(len(target_nodes), len(reg_df)),
        ).fit()

    ranked_clusters = cluster_summary.sort_values(
        by=["target_breadth", "feed_count", "target_total_count", "total_count"],
        ascending=[False, False, False, False],
    )

    return {
        "settings": settings,
        "input_data": dat,
        "count_table": count_table,
        "cluster_summary": cluster_summary,
        "observed_stats": observed_stats,
        "perm_results": perm_results,
        "pval_table": pval_table,
        "regression_results": regression_results,
        "ranked_clusters": ranked_clusters,
    }


def make_qbins(x: pd.Series, q: int = 5) -> pd.Series:
    """Quantile bins with fallback for low-variation data."""
    x = pd.Series(x).astype(float)
    if x.nunique() <= 1:
        return pd.Series(["all"] * len(x), index=x.index)
    q_eff = min(q, x.nunique())
    try:
        bins = pd.qcut(x, q=q_eff, duplicates="drop")
    except ValueError:
        bins = pd.qcut(x.rank(method="average"), q=q_eff, duplicates="drop")
    return bins.astype(str)


def build_burden_cluster_summary(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    node_col: str,
    feed_node: str,
    human_node: str,
    animal_target_nodes: Sequence[str],
) -> pd.DataFrame:
    """Build cluster-level feed, animal-target, and human burden summaries."""
    _, count_table = build_cluster_count_table(
        df,
        cluster_col=cluster_col,
        node_col=node_col,
        feed_node=feed_node,
        target_nodes=animal_target_nodes,
        extra_nodes=[human_node],
    )
    summary = pd.DataFrame(index=count_table.index)
    summary["feed_count"] = count_table[feed_node]
    summary["feed_present"] = (summary["feed_count"] > 0).astype(int)
    summary["animal_target_breadth"] = (count_table[list(animal_target_nodes)] > 0).sum(axis=1)
    summary["animal_target_total_count"] = count_table[list(animal_target_nodes)].sum(axis=1)
    summary["human_count"] = count_table[human_node]
    summary["human_present"] = (summary["human_count"] > 0).astype(int)
    summary["total_count"] = count_table.sum(axis=1)
    summary["log_total_count"] = np.log1p(summary["total_count"])
    summary["global_breadth"] = (count_table > 0).sum(axis=1)
    summary["human_fraction"] = np.where(
        summary["total_count"] > 0,
        summary["human_count"] / summary["total_count"],
        np.nan,
    )
    return summary


def add_size_adjusted_human_metric(
    eligible: pd.DataFrame,
    residual_formula: str,
) -> tuple[pd.DataFrame, object]:
    """Add negative-binomial expected counts and residuals for human burden."""
    work = eligible.copy().reset_index(drop=False)
    model = smf.negativebinomial(residual_formula, data=work).fit(disp=False)
    fitted = np.asarray(model.predict(work), dtype=float)
    observed = work["human_count"].to_numpy(dtype=float)
    alpha = float(model.params["alpha"])
    variance = fitted + alpha * (fitted**2)
    work["human_expected"] = fitted
    work["human_excess_count"] = observed - fitted
    work["human_size_adjusted_resid"] = (observed - fitted) / np.sqrt(variance)
    out = work.set_index(eligible.index.name if eligible.index.name is not None else "index", drop=True)
    out.index = eligible.index
    return out, model


def define_human_top(
    eligible: pd.DataFrame,
    *,
    human_metric: str,
    top_fraction: float,
) -> pd.DataFrame:
    """Mark the top fraction of clusters by the selected human-burden metric."""
    out = eligible.copy()
    n_top = max(1, int(np.ceil(len(out) * top_fraction)))
    top_idx = out[human_metric].sort_values(ascending=False).head(n_top).index
    out["human_top"] = 0
    out.loc[top_idx, "human_top"] = 1
    return out


def odds_ratio_from_2x2(a: int, b: int, c: int, d: int, correction: float = 0.5) -> float:
    return ((a + correction) * (d + correction)) / ((b + correction) * (c + correction))


def compute_enrichment_stats(
    eligible: pd.DataFrame,
    *,
    eco_label_col: str,
    human_top_col: str = "human_top",
) -> dict[str, float]:
    top = eligible[human_top_col] == 1
    not_top = eligible[human_top_col] == 0
    eco = eligible[eco_label_col] == 1
    other = eligible[eco_label_col] == 0

    a = int((eco & top).sum())
    b = int((eco & not_top).sum())
    c = int((other & top).sum())
    d = int((other & not_top).sum())

    n_eco = int(eco.sum())
    n_other = int(other.sum())
    n_top = int(top.sum())

    return {
        "n_eligible_clusters": int(len(eligible)),
        "n_eco_subset": n_eco,
        "n_other": n_other,
        "n_top_human": n_top,
        "eco_in_top_human": a,
        "prop_top_human_that_are_eco": a / n_top if n_top else np.nan,
        "prop_eco_that_are_top_human": a / n_eco if n_eco else np.nan,
        "odds_ratio_eco_vs_top_human": odds_ratio_from_2x2(a, b, c, d),
        "mean_human_metric_eco": eligible.loc[eco, "human_metric_for_summary"].mean(),
        "mean_human_metric_other": eligible.loc[other, "human_metric_for_summary"].mean(),
        "mean_human_metric_diff": (
            eligible.loc[eco, "human_metric_for_summary"].mean()
            - eligible.loc[other, "human_metric_for_summary"].mean()
        ),
    }


def permute_eco_within_strata(
    eligible: pd.DataFrame,
    rng: np.random.Generator,
    *,
    eco_label_col: str = "eco_subset",
    stratum_col: str = "perm_stratum",
) -> pd.Series:
    """Shuffle ecological-subset labels within size-by-breadth strata."""
    permuted = pd.Series(index=eligible.index, dtype=int)
    for _, idx in eligible.groupby(stratum_col).groups.items():
        idx = list(idx)
        permuted.loc[idx] = rng.permutation(eligible.loc[idx, eco_label_col].to_numpy(copy=True))
    return permuted.astype(int)


def run_top_lineage_enrichment_test(
    df: pd.DataFrame,
    settings: HumanBurdenSettings = HumanBurdenSettings(),
) -> dict[str, object]:
    """Test whether feed-linked broad animal clusters are enriched among top human-burden clusters."""
    rng = np.random.default_rng(settings.random_seed)
    target_nodes = list(settings.animal_target_nodes)
    cluster_summary = build_burden_cluster_summary(
        df,
        cluster_col=settings.cluster_col,
        node_col=settings.node_col,
        feed_node=settings.feed_node,
        human_node=settings.human_node,
        animal_target_nodes=target_nodes,
    )

    eligible = cluster_summary.loc[
        (cluster_summary["animal_target_breadth"] >= settings.min_animal_breadth)
        & (cluster_summary["animal_target_total_count"] >= settings.min_animal_target_total_count)
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible HC clusters after applying animal-breadth criteria.")

    eligible["eco_subset"] = (eligible["feed_present"] == 1).astype(int)
    eligible, nb_model = add_size_adjusted_human_metric(eligible, settings.residual_formula)
    if settings.human_metric not in eligible.columns:
        raise ValueError(f"Unknown human metric: {settings.human_metric}")

    eligible = define_human_top(
        eligible,
        human_metric=settings.human_metric,
        top_fraction=settings.top_fraction,
    )
    eligible["human_metric_for_summary"] = eligible[settings.human_metric]
    eligible["size_bin"] = make_qbins(eligible["log_total_count"], q=settings.n_size_bins)
    eligible["perm_stratum"] = eligible["animal_target_breadth"].astype(str) + " | " + eligible["size_bin"]

    observed_stats = compute_enrichment_stats(eligible, eco_label_col="eco_subset")
    perm_rows = []
    for _ in range(settings.n_perm):
        perm_eligible = eligible.copy()
        perm_eligible["perm_eco_subset"] = permute_eco_within_strata(
            eligible,
            rng,
            eco_label_col="eco_subset",
            stratum_col="perm_stratum",
        )
        perm_rows.append(compute_enrichment_stats(perm_eligible, eco_label_col="perm_eco_subset"))

    perm_results = pd.DataFrame(perm_rows)
    pval_table = pd.DataFrame(
        {
            stat: {
                "observed": observed_stats.get(stat, np.nan),
                "perm_mean": perm_results[stat].mean(skipna=True),
                "perm_sd": perm_results[stat].std(skipna=True),
                "p_one_sided_greater": empirical_p_greater(observed_stats.get(stat, np.nan), perm_results[stat]),
                "p_two_sided": empirical_p_two_sided(observed_stats.get(stat, np.nan), perm_results[stat]),
            }
            for stat in perm_results.columns
        }
    ).T

    regression_results = {}
    reg_df = eligible.reset_index(drop=True)
    try:
        if reg_df["human_top"].nunique() > 1 and reg_df["eco_subset"].nunique() > 1:
            regression_results["logit_human_top_model"] = smf.logit(
                "human_top ~ eco_subset + log_total_count + animal_target_breadth",
                data=reg_df,
            ).fit(disp=False)
    except Exception as exc:  # pragma: no cover - depends on data separation
        regression_results["logit_human_top_model_error"] = str(exc)

    ranked_eligible = eligible.sort_values(
        by=["human_top", settings.human_metric, "eco_subset", "human_count"],
        ascending=[False, False, False, False],
    )

    return {
        "settings": settings,
        "cluster_summary": cluster_summary,
        "eligible_set": eligible,
        "observed_stats": observed_stats,
        "perm_results": perm_results,
        "pval_table": pval_table,
        "baseline_nb_model": nb_model,
        "regression_results": regression_results,
        "ranked_eligible": ranked_eligible,
    }


def table_from_binary_vectors(exposure: pd.Series, outcome: pd.Series) -> np.ndarray:
    exposure = pd.Series(exposure).astype(int)
    outcome = pd.Series(outcome).astype(int)
    return np.array(
        [
            [int(((exposure == 1) & (outcome == 1)).sum()), int(((exposure == 1) & (outcome == 0)).sum())],
            [int(((exposure == 0) & (outcome == 1)).sum()), int(((exposure == 0) & (outcome == 0)).sum())],
        ],
        dtype=int,
    )


def run_cmh_feed_human_presence_test(
    df: pd.DataFrame,
    settings: HumanBurdenSettings = HumanBurdenSettings(),
    *,
    add_haldane_correction_to_zero_strata: bool = True,
) -> dict[str, object]:
    """Stratified human-presence test for broad feed-associated animal clusters."""
    target_nodes = list(settings.animal_target_nodes)
    cluster_summary = build_burden_cluster_summary(
        df,
        cluster_col=settings.cluster_col,
        node_col=settings.node_col,
        feed_node=settings.feed_node,
        human_node=settings.human_node,
        animal_target_nodes=target_nodes,
    )
    eligible = cluster_summary.loc[
        (cluster_summary["animal_target_breadth"] >= settings.min_animal_breadth)
        & (cluster_summary["animal_target_total_count"] >= settings.min_animal_target_total_count)
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible HC clusters after applying animal-breadth criteria.")

    eligible["eco_subset"] = (eligible["feed_present"] == 1).astype(int)
    eligible["size_bin"] = make_qbins(eligible["log_total_count"], q=settings.n_size_bins)
    eligible["cmh_stratum"] = eligible["animal_target_breadth"].astype(str) + " | " + eligible["size_bin"]

    crude_table = table_from_binary_vectors(eligible["eco_subset"], eligible["human_present"])
    fisher_or, fisher_p_two_sided = fisher_exact(crude_table, alternative="two-sided")
    _, fisher_p_greater = fisher_exact(crude_table, alternative="greater")
    chi2_stat, chi2_p, _, expected = chi2_contingency(crude_table, correction=False)

    stratum_rows = []
    stratified_tables = []
    for stratum, sub in eligible.groupby("cmh_stratum"):
        table = table_from_binary_vectors(sub["eco_subset"], sub["human_present"])
        stratum_rows.append(
            {
                "stratum": stratum,
                "n_clusters": int(len(sub)),
                "n_eco_subset": int(sub["eco_subset"].sum()),
                "n_other": int((sub["eco_subset"] == 0).sum()),
                "n_human_present": int(sub["human_present"].sum()),
                "a_eco_and_human": int(table[0, 0]),
                "b_eco_and_not_human": int(table[0, 1]),
                "c_other_and_human": int(table[1, 0]),
                "d_other_and_not_human": int(table[1, 1]),
            }
        )
        if add_haldane_correction_to_zero_strata and np.any(table == 0):
            stratified_tables.append(table.astype(float) + 0.5)
        else:
            stratified_tables.append(table.astype(float))

    cmh = StratifiedTable(stratified_tables)
    cmh_test = cmh.test_null_odds()
    pooled_or_ci = cmh.oddsratio_pooled_confint()

    return {
        "settings": settings,
        "cluster_summary": cluster_summary,
        "eligible_set": eligible,
        "crude_table": crude_table,
        "crude_table_df": pd.DataFrame(
            crude_table,
            index=["eco_subset=1", "eco_subset=0"],
            columns=["human_present=1", "human_present=0"],
        ),
        "crude_tests": {
            "fisher_odds_ratio": fisher_or,
            "fisher_p_two_sided": fisher_p_two_sided,
            "fisher_p_one_sided_greater": fisher_p_greater,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "expected_counts": expected,
        },
        "stratum_tables": pd.DataFrame(stratum_rows),
        "cmh_results": {
            "pooled_odds_ratio": cmh.oddsratio_pooled,
            "pooled_odds_ratio_ci_low": pooled_or_ci[0],
            "pooled_odds_ratio_ci_high": pooled_or_ci[1],
            "cmh_statistic": cmh_test.statistic,
            "cmh_p_two_sided": cmh_test.pvalue,
        },
    }


def write_table(df: pd.DataFrame, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    df.to_csv(path)
    return path


def run_cli(args: argparse.Namespace) -> None:
    df = read_metadata(args.input)
    if args.add_feed_subclass:
        df = add_animal_feed_subclass(df, node_col=args.node_col, feed_node=args.feed_node)

    feed_settings = FeedBreadthSettings(
        cluster_col=args.cluster_col,
        node_col=args.node_col,
        feed_node=args.feed_node,
        target_nodes=tuple(args.target_nodes),
        n_perm=args.n_perm,
        random_seed=args.random_seed,
    )
    feed_results = run_feed_breadth_test(df, feed_settings)
    print("\n[Feed breadth observed statistics]")
    print(pd.Series(feed_results["observed_stats"]).to_string())
    print("\n[Feed breadth permutation summary]")
    print(feed_results["pval_table"].to_string())

    if args.output_dir:
        output_dir = Path(args.output_dir)
        write_table(feed_results["cluster_summary"], output_dir, "feed_breadth_cluster_summary.csv")
        write_table(feed_results["pval_table"], output_dir, "feed_breadth_permutation_summary.csv")
        write_table(feed_results["ranked_clusters"], output_dir, "feed_breadth_ranked_clusters.csv")

    if args.skip_human_burden:
        return

    for subset in normalize_subset_names(args.human_subsets):
        df_subset = filter_by_human_attribution(
            df,
            None if subset == "all" else subset,
            node_col=args.node_col,
            human_label=args.human_node,
            dist_cutoff=args.dist_cutoff,
            trust_cutoff=args.trust_cutoff,
        )
        burden_settings = HumanBurdenSettings(
            cluster_col=args.cluster_col,
            node_col=args.node_col,
            feed_node=args.feed_node,
            human_node=args.human_node,
            animal_target_nodes=tuple(args.target_nodes),
            min_animal_breadth=args.min_animal_breadth,
            min_animal_target_total_count=args.min_animal_target_total_count,
            top_fraction=args.top_fraction,
            n_size_bins=args.n_size_bins,
            n_perm=args.human_n_perm,
            random_seed=args.human_random_seed,
        )
        burden_results = run_top_lineage_enrichment_test(df_subset, burden_settings)
        print(f"\n[Human-burden enrichment: subset={subset}]")
        print(pd.Series(burden_results["observed_stats"]).to_string())
        print(burden_results["pval_table"].to_string())

        cmh_results = run_cmh_feed_human_presence_test(df_subset, burden_settings)
        print(f"\n[Human-presence CMH: subset={subset}]")
        print(pd.Series(cmh_results["cmh_results"]).to_string())

        if args.output_dir:
            safe_subset = subset.lower()
            output_dir = Path(args.output_dir)
            write_table(burden_results["eligible_set"], output_dir, f"human_burden_eligible_{safe_subset}.csv")
            write_table(burden_results["pval_table"], output_dir, f"human_burden_permutation_summary_{safe_subset}.csv")
            write_table(cmh_results["crude_table_df"], output_dir, f"human_presence_crude_table_{safe_subset}.csv")
            write_table(cmh_results["stratum_tables"], output_dir, f"human_presence_cmh_strata_{safe_subset}.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_PATH / "metadata_training_testing.csv"), help="Metadata CSV.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV result tables.")
    parser.add_argument("--cluster-col", default="HC50")
    parser.add_argument("--node-col", default="curated_source_region")
    parser.add_argument("--feed-node", default="animal_feed")
    parser.add_argument("--human-node", default="human")
    parser.add_argument("--target-nodes", nargs="+", default=DEFAULT_TARGET_NODES)
    parser.add_argument("--n-perm", type=int, default=2000, help="Permutations for the feed-breadth test.")
    parser.add_argument("--random-seed", type=int, default=186)
    parser.add_argument("--add-feed-subclass", action="store_true", help="Add pet-food/non-pet-food feed labels before analysis.")
    parser.add_argument("--skip-human-burden", action="store_true")
    parser.add_argument("--human-subsets", nargs="+", default=["all"], help="Human subsets: all, F, G, FG.")
    parser.add_argument("--human-n-perm", type=int, default=10000)
    parser.add_argument("--human-random-seed", type=int, default=42)
    parser.add_argument("--min-animal-breadth", type=int, default=3)
    parser.add_argument("--min-animal-target-total-count", type=int, default=0)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--n-size-bins", type=int, default=5)
    parser.add_argument("--dist-cutoff", type=float, default=DEFAULT_DIST_CUTOFF)
    parser.add_argument("--trust-cutoff", type=float, default=DEFAULT_TRUST_CUTOFF)
    return parser


if __name__ == "__main__":
    run_cli(build_arg_parser().parse_args())
