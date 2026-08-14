import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

# ============================================================
# COLOR MAPPING
# ============================================================
color_dir = {
    'bovine':'lightskyblue',
    'swine':'slateblue',
    'poultry':'gold'
}

def get_color(source):
    low = str(source).lower()
    for k, v in color_dir.items():
        if k.lower() == low:
            return v
    return 'grey'

# ============================================================
# CORE METRICS
# ============================================================
def compute_ecdf(distances, max_d=100):
    xs = np.arange(0, max_d + 1)
    ys = np.array([(distances <= d).mean() for d in xs])
    return xs, ys

def ecdf_auc(xs, ys, d_max=50):
    mask = xs <= d_max
    return np.trapezoid(ys[mask], xs[mask]) / d_max

def goods_coverage_from_counts(counts):
    counts = np.asarray(counts, dtype=int)
    N = counts.sum()
    f1 = np.sum(counts == 1)
    return 1 - f1 / N if N > 0 else np.nan

def distance_coverage_from_series(dists, threshold=50):
    if isinstance(dists, np.ndarray):
        d = pd.Series(dists)
    else:
        d = dists
    d = pd.to_numeric(d, errors='coerce').dropna()
    N = len(d)
    if N == 0:
        return np.nan
    return 1 - (d.gt(threshold).sum() / N)

def chao1(counts):
    counts = np.asarray(counts, dtype=int)
    S_obs = np.sum(counts > 0)
    f1 = np.sum(counts == 1)
    f2 = np.sum(counts == 2)
    if f2 > 0:
        est = S_obs + (f1 * f1) / (2 * f2)
    else:
        est = S_obs + (f1 * (f1 - 1)) / 2.0
    return est, S_obs

def auto_rarefaction_cutoff(df, quantile=0.1, min_cutoff=50):
    Ns = df.groupby("Source").size()
    n_star = int(max(min_cutoff, np.quantile(Ns, quantile)))
    print(f"[INFO] Using n* = {n_star} isolates per source (10th percentile).")
    return n_star

# ============================================================
# RAREFIED COMPUTATION
# ============================================================
def rarefied_ecdf(
    df,
    lineage_col="HC50",
    distance_col="Closest distance",
    thresholds=(20, 50, 100),
    n_star=150,
    B=500,
    random_seed=42,
    min_isolates=30,
    max_d=100,
    auc_dmax=50,
    ci=(2.5, 97.5),
    min_dist_points=5
):
    rng = np.random.default_rng(random_seed)
    xs = np.arange(0, max_d + 1)
    auc_mask = xs <= auc_dmax
    xs_auc = xs[auc_mask]

    results_ecdf = {}
    summary_rows = []

    def summarize(arr):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return dict(mean=np.nan, lo=np.nan, hi=np.nan)
        return dict(
            mean=float(np.mean(arr)),
            lo=float(np.percentile(arr, ci[0])),
            hi=float(np.percentile(arr, ci[1]))
        )

    for src, sub in df.groupby("Source"):
        N = len(sub)
        if N < max(n_star, min_isolates):
            continue

        lin = sub[lineage_col].astype(str).values
        dists = pd.to_numeric(sub[distance_col], errors="coerce").values
        idx_all = np.arange(N)

        # ---- store per-bootstrap ECDF curve + scalar metrics
        ecdfs = np.full((B, xs.size), np.nan, dtype=float)
        auc_vals = np.full(B, np.nan, dtype=float)
        goods_vals = np.full(B, np.nan, dtype=float)
        obs_chao1 = np.full(B, np.nan, dtype=float)
        distcov_vals = {thr: np.full(B, np.nan, dtype=float) for thr in thresholds}

        for b in range(B):
            idx = rng.choice(idx_all, size=n_star, replace=False)

            # distances
            dsub = dists[idx]
            dsub = dsub[~np.isnan(dsub)]

            # lineage counts
            counts = pd.Series(lin[idx]).value_counts().values
            goods_vals[b] = goods_coverage_from_counts(counts)
            est, S_obs = chao1(counts)
            obs_chao1[b] = S_obs / est if est > 0 else np.nan

            for thr in thresholds:
                distcov_vals[thr][b] = distance_coverage_from_series(dsub, threshold=thr)

            # ECDF curve + AUC (same draw)
            if len(dsub) < min_dist_points:
                continue
            y = (dsub[:, None] <= xs[None, :]).mean(axis=0)
            ecdfs[b, :] = y
            auc_vals[b] = np.trapezoid(y[auc_mask], xs_auc) / auc_dmax

        # use only successful draws (those that have ECDF + AUC)
        ok = np.isfinite(ecdfs).all(axis=1) & np.isfinite(auc_vals)
        ecdfs_ok = ecdfs[ok, :]
        auc_ok = auc_vals[ok]
        if ecdfs_ok.shape[0] == 0:
            continue

        # ---- ECDF results for plotting
        results_ecdf[str(src).lower()] = {
            "xs": xs,
            "mean": np.mean(ecdfs_ok, axis=0),
            "lower": np.percentile(ecdfs_ok, ci[0], axis=0),
            "upper": np.percentile(ecdfs_ok, ci[1], axis=0),
            "n_original": int(N),
            "n_star": int(n_star),
            "B_eff": int(ecdfs_ok.shape[0]),
        }

        # ---- summary table
        row = {
            "Source": str(src).lower(),
            "N available": int(N),
            "n*": int(n_star),
            "B_eff": int(ecdfs_ok.shape[0]),
        }
        auc_sum = summarize(auc_ok)
        row.update({"AUC_mean": auc_sum["mean"], "AUC_lo": auc_sum["lo"], "AUC_hi": auc_sum["hi"]})
        goods_sum = summarize(goods_vals)
        row.update({"GoodsC_mean": goods_sum["mean"], "GoodsC_lo": goods_sum["lo"], "GoodsC_hi": goods_sum["hi"]})
        obs_sum = summarize(obs_chao1)
        row.update({"ObsChao1_mean": obs_sum["mean"], "ObsChao1_lo": obs_sum["lo"], "ObsChao1_hi": obs_sum["hi"]})
        for thr, vals in distcov_vals.items():
            s = summarize(vals)
            row.update({f"DistC{thr}_mean": s["mean"], f"DistC{thr}_lo": s["lo"], f"DistC{thr}_hi": s["hi"]})
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("AUC_mean", ascending=False)
    return results_ecdf, summary_df


def plot_ecdf_multiples(
    results,
    auc_df,
    ncols=3,
    title="ECDFs by Source",
    auc_dmax=50,
    area_alpha=0.22,
    area_color="lightgray",
    show_auc_ci=True,
    # title options
    show_subtitle=True,
    show_subplot_title=True,
    label_override=None,
    # font sizes
    tick_label_fontsize=10,
    axis_label_fontsize=11,
    title_label_fontsize=12,
    # axis label display
    hide_axis_label=False,
    # frame
    frame_weight=1.0,
    # panel geometry
    panel_ratio=4/3,
    # color options
    line_color=None,
    ci_color=None,
    ci_alpha=0.15
):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    tmp = auc_df.copy()
    tmp["Source"] = tmp["Source"].astype(str).str.lower()
    auc_map = {
        s: (m, lo, hi)
        for s, m, lo, hi in zip(tmp["Source"], tmp["AUC_mean"], tmp["AUC_lo"], tmp["AUC_hi"])
    }

    def _resolve_color(spec, src, fallback_func):
        if spec is None:
            return fallback_func(src)
        if isinstance(spec, dict):
            return spec.get(src, spec.get(str(src).lower(), fallback_func(src)))
        return spec

    def _resolve_label(src, label_override):
        if not isinstance(label_override, dict):
            return src
        return label_override.get(src, label_override.get(str(src).lower(), src))

    def _round_up_2(x):
        if not np.isfinite(x):
            return np.nan
        return math.ceil(float(x) * 100) / 100.0

    auc_lookup = tmp.set_index("Source")["AUC_mean"]
    src_df = pd.DataFrame({"src": list(results.keys())})
    src_df["src_lower"] = src_df["src"].astype(str).str.lower()
    src_df["AUC_mean"] = src_df["src_lower"].map(auc_lookup)
    src_df = src_df.sort_values(
        by=["AUC_mean", "src_lower"],
        ascending=[False, True],
        na_position="last"
    )
    srcs = src_df["src"].tolist()

    n = len(srcs)
    if n == 0:
        raise ValueError("results is empty.")

    nrows = math.ceil(n / ncols)

    panel_width = 4
    panel_height = panel_width / panel_ratio

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
        dpi=600
    )

    for i, src in enumerate(srcs):
        r = results[src]
        row_idx, col_idx = divmod(i, ncols)
        ax = axes[row_idx, col_idx]

        lc = _resolve_color(line_color, src, get_color)
        cc = _resolve_color(ci_color, src, (lambda s: lc))

        xs = np.asarray(r["xs"])
        mean = np.asarray(r["mean"])
        lower = np.asarray(r["lower"])
        upper = np.asarray(r["upper"])

        ax.plot(xs, mean, color=lc, lw=2, zorder=3)
        ax.fill_between(xs, lower, upper, color=cc, alpha=ci_alpha, zorder=2)

        mask = xs <= auc_dmax
        ax.fill_between(
            xs[mask], 0, mean[mask],
            color=area_color, alpha=area_alpha,
            linewidth=0, zorder=1
        )
        ax.axvline(auc_dmax, color="gray", lw=1, ls="--", alpha=0.5)

        auc_m, auc_lo, auc_hi = auc_map.get(str(src).lower(), (np.nan, np.nan, np.nan))
        if np.isfinite(auc_m) and show_auc_ci and np.isfinite(auc_lo) and np.isfinite(auc_hi):
            auc_txt = f"AUC={auc_m:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]"
        elif np.isfinite(auc_m):
            auc_txt = f"AUC={auc_m:.3f}"
        else:
            auc_txt = "AUC=NA"

        display_label = _resolve_label(src, label_override)

        if show_subplot_title:
            ax.set_title(
                f"{display_label} (n={r.get('n_original', 'NA')}) | {auc_txt}",
                fontsize=title_label_fontsize
            )
        else:
            # Lower-right in-panel label block with padding from frame
            x_text = 0.965
            y_auc = 0.040
            y_label = 0.125

            auc_up = _round_up_2(auc_m)
            auc_inside_txt = f"AUC={auc_up:.2f}" if np.isfinite(auc_up) else "AUC=NA"

            ax.text(
                x_text, y_label,
                display_label,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=axis_label_fontsize + 2,
                color="black",
                zorder=4
            )
            ax.text(
                x_text, y_auc,
                auc_inside_txt,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=axis_label_fontsize + 1,
                color="black",
                zorder=4
            )

        if hide_axis_label:
            ax.set_xlabel(
                "Allelic differences" if row_idx == nrows - 1 else "",
                fontsize=axis_label_fontsize
            )
            ax.set_ylabel(
                "Proportion" if col_idx == 0 else "",
                fontsize=axis_label_fontsize
            )
        else:
            ax.set_xlabel("Allelic differences", fontsize=axis_label_fontsize)
            ax.set_ylabel("Proportion", fontsize=axis_label_fontsize)

        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        for sp in ax.spines.values():
            sp.set_linewidth(frame_weight)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if show_subtitle:
        fig.suptitle(title, fontsize=title_label_fontsize)

    fig.tight_layout()
    return fig, axes


## Plot diversity vs coverage metrics
def plot_diversity_vs_coverage(
    df,
    y_metric="AUC",
    x_metric="GoodsC",
    threshold=100,
    base_size=1,
    size_power=0.9,
    node_size_range=(0.0, 800.0),
    annotate=True,
    node_label=None,
    node_label_fontsize=8.5,
    axis_label_fontsize=11,
    frame_weight=1.2,
    tick_fontsize=10,
    cmap_func=get_color,
    regression="none",
    band_B=500,
    random_seed=186,
    line_color="firebrick",
    band_alpha=0.08,
    quadrant=False,
    quadrant_x=None,
    quadrant_y=None,
    color_by_ObsChao1=False,
    cmap_name="viridis",
    xlim=(0.5, 0.9),
    ylim=(0.1, 0.7),
    figsize=(8, 6),
    min_node_size=None,
    label_pad_pts=1.0,
    label_pos_override=None,   # {'fruits_vegetables': (1, 0.5)}
    # errorbar style
    errorbar_color="gray",
    errorbar_alpha=0.6,
    errorbar_lw=1.0,
    errorbar_capsize=2.0,
    # node style
    node_alpha=0.85,
    node_edge_color="k",
    node_edge_weight=0.5,
):
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if node_label is not None and not isinstance(node_label, dict):
        raise ValueError("node_label must be None or dict.")
    if not (isinstance(figsize, (tuple, list)) and len(figsize) == 2):
        raise ValueError("figsize must be (w, h).")
    if min_node_size is not None and float(min_node_size) < 0:
        raise ValueError("min_node_size must be >= 0.")
    if float(label_pad_pts) < 0:
        raise ValueError("label_pad_pts must be >= 0.")

    if not (isinstance(node_size_range, (tuple, list)) and len(node_size_range) == 2):
        raise ValueError("node_size_range must be (min_size, max_size).")
    node_size_min, node_size_max = float(node_size_range[0]), float(node_size_range[1])
    if node_size_min < 0 or node_size_max < 0 or node_size_min > node_size_max:
        raise ValueError("node_size_range must satisfy 0 <= min_size <= max_size.")

    if not (0 <= float(node_alpha) <= 1):
        raise ValueError("node_alpha must be in [0, 1].")
    if float(node_edge_weight) < 0:
        raise ValueError("node_edge_weight must be >= 0.")
    if float(errorbar_alpha) < 0 or float(errorbar_alpha) > 1:
        raise ValueError("errorbar_alpha must be in [0, 1].")
    if float(errorbar_lw) < 0:
        raise ValueError("errorbar_lw must be >= 0.")
    if float(errorbar_capsize) < 0:
        raise ValueError("errorbar_capsize must be >= 0.")

    def _norm_name(x):
        s = str(x).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    override_map = {}
    if label_pos_override is not None:
        if not isinstance(label_pos_override, dict):
            raise ValueError("label_pos_override must be None or dict.")
        for k, v in label_pos_override.items():
            if not (isinstance(v, (tuple, list)) and len(v) == 2):
                raise ValueError(f"label_pos_override[{k}] must be (pos_code, pad_pts).")
            pos_code, pad = int(v[0]), float(v[1])
            if pos_code not in (-1, 0, 1):
                raise ValueError(f"label_pos_override[{k}] pos_code must be -1, 0, or 1.")
            if pad < 0:
                raise ValueError(f"label_pos_override[{k}] pad must be >= 0.")
            override_map[_norm_name(k)] = (pos_code, pad)

    plot_df = df.copy()
    if min_node_size is not None:
        if "N available" not in plot_df.columns:
            raise ValueError("df must contain 'N available' when min_node_size is used.")
        plot_df = plot_df.loc[plot_df["N available"] >= float(min_node_size)].copy()
    if plot_df.empty:
        raise ValueError("No rows left after min_node_size filter.")

    ykey = y_metric.strip().lower()
    if ykey == "auc":
        y_mean_col, y_lo_col, y_hi_col = "AUC_mean", "AUC_lo", "AUC_hi"
        y_transform = lambda m, lo, hi: (m, lo, hi)
        ylabel = "Mean AUC (compactness)"
    elif ykey in ("1-auc", "inv_auc", "inverse_auc", "diversity"):
        y_mean_col, y_lo_col, y_hi_col = "AUC_mean", "AUC_lo", "AUC_hi"
        y_transform = lambda m, lo, hi: (1.0 - m, 1.0 - hi, 1.0 - lo)
        ylabel = "1 − Mean AUC (apparent within-node phylogenetic dispersion)"
    else:
        y_mean_col, y_lo_col, y_hi_col = "ObsChao1_mean", "ObsChao1_lo", "ObsChao1_hi"
        y_transform = lambda m, lo, hi: (m, lo, hi)
        ylabel = "Observed / Chao1 richness ratio"

    xkey = x_metric.strip().lower()
    if xkey.startswith("good"):
        x_mean_col, x_lo_col, x_hi_col = "GoodsC_mean", "GoodsC_lo", "GoodsC_hi"
        xlabel = "Good's C (coverage)"
    elif xkey.startswith("dist"):
        x_mean_col = f"DistC{threshold}_mean"
        x_lo_col = f"DistC{threshold}_lo"
        x_hi_col = f"DistC{threshold}_hi"
        xlabel = f"Distance-based coverage (≤{threshold} allelic differences)"
    elif xkey.startswith("obs"):
        x_mean_col, x_lo_col, x_hi_col = "ObsChao1_mean", "ObsChao1_lo", "ObsChao1_hi"
        xlabel = "Observed / Chao1 richness ratio"
    else:
        raise ValueError("x_metric must be 'GoodsC', 'DistC', or 'ObsChao1'.")

    fig, ax = plt.subplots(figsize=figsize, dpi=600)

    Z_BAND, Z_LINE, Z_ERR, Z_SCAT, Z_TEXT = 0.0, 0.5, 2.0, 3.0, 4.0

    nvals = pd.to_numeric(plot_df["N available"], errors="coerce").astype(float)
    nmin, nmax = float(np.nanmin(nvals)), float(np.nanmax(nvals))
    if np.isclose(nmin, nmax):
        sizes = pd.Series(np.full(len(plot_df), (node_size_min + node_size_max) / 2.0), index=plot_df.index)
    else:
        scaled = (nvals ** size_power - nmin ** size_power) / (nmax ** size_power - nmin ** size_power)
        sizes = node_size_min + (node_size_max - node_size_min) * scaled

    cmap = plt.get_cmap(cmap_name)
    norm = None
    if color_by_ObsChao1:
        vals = pd.to_numeric(
            plot_df.get("ObsChao1_mean", pd.Series(index=plot_df.index, dtype=float)),
            errors="coerce",
        )
        vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        if vmin == vmax:
            vmin, vmax = max(0.0, vmin - 1e-6), vmin + 1e-6
        vmin = min(vmin, 0.1)
        vmax = max(vmax, 0.5)

        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _resolve_label(src):
        if node_label is None:
            return src
        if src in node_label:
            return node_label[src]
        src_l = str(src).lower()
        for k, v in node_label.items():
            if str(k).lower() == src_l:
                return v
        return src

    Xs, Ys = [], []

    for k, r in enumerate(plot_df.itertuples(index=False)):
        x_mean, x_lo, x_hi = getattr(r, x_mean_col), getattr(r, x_lo_col), getattr(r, x_hi_col)
        y_mean_raw, y_lo_raw, y_hi_raw = getattr(r, y_mean_col), getattr(r, y_lo_col), getattr(r, y_hi_col)
        y_mean, y_lo, y_hi = y_transform(y_mean_raw, y_lo_raw, y_hi_raw)
        src = getattr(r, "Source")

        if color_by_ObsChao1 and norm is not None:
            obs_ratio = getattr(r, "ObsChao1_mean", np.nan)
            color = cmap(norm(obs_ratio)) if np.isfinite(obs_ratio) else (0.6, 0.6, 0.6, 1.0)
        else:
            color = cmap_func(src)

        s = float(sizes.iloc[k])

        ax.scatter(
            x_mean, y_mean,
            s=s,
            color=color,
            edgecolor=node_edge_color,
            lw=node_edge_weight,
            alpha=node_alpha,
            zorder=Z_SCAT,
        )

        ax.errorbar(
            x_mean, y_mean,
            xerr=[[x_mean - x_lo], [x_hi - x_mean]],
            yerr=[[y_mean - y_lo], [y_hi - y_mean]],
            fmt="none",
            ecolor=errorbar_color,
            alpha=errorbar_alpha,
            elinewidth=errorbar_lw,
            capsize=errorbar_capsize,
            zorder=Z_ERR,
        )

        if annotate:
            shown = _resolve_label(src)

            r_pts = np.sqrt(max(s, 0.0) / np.pi) + float(node_edge_weight) / 2.0
            src_key = _norm_name(src)
            shown_key = _norm_name(shown)
            pos_code, pad = override_map.get(src_key, override_map.get(shown_key, (1, float(label_pad_pts))))
            d = r_pts + pad

            if pos_code == 1:      # upper-right
                dx_pts, dy_pts = d / np.sqrt(2.0), d / np.sqrt(2.0)
                ha, va = "left", "bottom"
            elif pos_code == 0:    # right horizontal
                dx_pts, dy_pts = d, 0.0
                ha, va = "left", "center"
            else:                  # -1 lower-right
                dx_pts, dy_pts = d / np.sqrt(2.0), -d / np.sqrt(2.0)
                ha, va = "left", "top"

            ax.annotate(
                shown,
                xy=(x_mean, y_mean),
                xycoords="data",
                xytext=(dx_pts, dy_pts),
                textcoords="offset points",
                ha=ha, va=va,
                fontsize=node_label_fontsize,
                zorder=Z_TEXT,
                clip_on=False,
            )

        Xs.append(x_mean)
        Ys.append(y_mean)

    if color_by_ObsChao1 and norm is not None:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.015)
        cbar.set_label("Obs/Chao1", fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize, width=frame_weight)

        # --- ONLY CHANGE 2: force exact ticks/labels on the colorbar
        ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

    if quadrant and len(Xs) and len(Ys):
        x_cut = quadrant_x if quadrant_x is not None else float(np.nanpercentile(Xs, 75))
        y_cut = quadrant_y if quadrant_y is not None else float(np.nanpercentile(Ys, 75))
        ax.axvline(x_cut, color="gray", lw=1.2, ls="--", alpha=0.7)
        ax.axhline(y_cut, color="gray", lw=1.2, ls="--", alpha=0.7)

    regression = (regression or "none").lower()
    if regression in ("linear", "huber") and len(Xs) >= 2:
        X = np.asarray(Xs)
        Y = np.asarray(Ys)
        x_grid = np.linspace(max(0.0, X.min() - 0.02), min(1.0, X.max() + 0.02), 300)

        rng = np.random.default_rng(random_seed)
        preds = np.zeros((band_B, x_grid.size))

        if regression == "linear":
            slope, intercept = np.polyfit(X, Y, 1)
            y_fit = slope * x_grid + intercept
            for b in range(band_B):
                idx = rng.integers(0, len(X), size=len(X))
                sb, tb = X[idx], Y[idx]
                m, c = np.polyfit(sb, tb, 1)
                preds[b] = m * x_grid + c
        else:
            try:
                from sklearn.linear_model import HuberRegressor
                hub = HuberRegressor().fit(X.reshape(-1, 1), Y)
                y_fit = hub.predict(x_grid.reshape(-1, 1)).ravel()
                for b in range(band_B):
                    idx = rng.integers(0, len(X), size=len(X))
                    sb, tb = X[idx].reshape(-1, 1), Y[idx]
                    try:
                        hb = HuberRegressor().fit(sb, tb)
                        preds[b] = hb.predict(x_grid.reshape(-1, 1)).ravel()
                    except Exception:
                        preds[b] = y_fit
            except Exception:
                slope, intercept = np.polyfit(X, Y, 1)
                y_fit = slope * x_grid + intercept
                for b in range(band_B):
                    idx = rng.integers(0, len(X), size=len(X))
                    sb, tb = X[idx], Y[idx]
                    m, c = np.polyfit(sb, tb, 1)
                    preds[b] = m * x_grid + c

        y_lo = np.percentile(preds, 2.5, axis=0)
        y_hi = np.percentile(preds, 97.5, axis=0)
        ax.fill_between(x_grid, y_lo, y_hi, color=line_color, alpha=band_alpha, linewidth=0, zorder=Z_BAND)
        ax.plot(x_grid, y_fit, color=line_color, lw=1.2, zorder=Z_LINE)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # extra right/top padding so upper-right labels like Poultry do not touch the frame
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = x1 - x0
    yr = y1 - y0
    ax.set_xlim(x0, x1 + 0.06 * xr)
    ax.set_ylim(y0, y1 + 0.02 * yr)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(frame_weight)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize, width=frame_weight)

    leg = ax.get_legend()
    if leg is not None:
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(axis_label_fontsize)
        for txt in leg.get_texts():
            txt.set_fontsize(tick_fontsize)

    plt.tight_layout()
    plt.show()
