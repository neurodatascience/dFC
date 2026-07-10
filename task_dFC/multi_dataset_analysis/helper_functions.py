import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors

# Curated palette of maximally distinct, publication-quality colors.
# Used for coloring high-performing experiments so each gets a clearly
# different hue even when only a few experiments are highlighted.
_VIBRANT_DISTINCT_COLORS = [
    "#E6194B",  # vivid red
    "#3CB44B",  # vivid green
    "#4363D8",  # vivid blue
    "#F58231",  # vivid orange
    "#911EB4",  # vivid purple
    "#42D4F4",  # cyan
    "#F032E6",  # magenta
    "#008080",  # teal
    "#9A6324",  # brown
    "#000075",  # navy
    "#808000",  # olive
    "#DC143C",  # crimson
]

###################### Publication style ######################


def setup_pub_style():
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update(
        {
            # Fonts & text
            "font.size": 18,  # base
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            # Lines/markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            # Figure/layout
            "figure.dpi": 150,  # on-screen
            "savefig.dpi": 1000,  # export
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            # Vector export: keep text as text in PDF/SVG
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig_pub(path_png_or_pdf: str):
    Path(Path(path_png_or_pdf).parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_png_or_pdf)
    # # Also export vector PDF alongside PNG unless you passed a .pdf
    # p = Path(path_png_or_pdf)
    # if p.suffix.lower() != ".pdf":
    #     plt.savefig(p.with_suffix(".pdf"))


###################### RDoC ######################

RDoC_MAP = {
    "real": {
        # --- Cognitive-Atlas–aligned domains (order on paper) ---
        "DOMAIN_ORDER": [
            "Arousal & Regulatory Systems",
            "Cognitive Systems",
            "Negative Valence System",
            "Positive Valence System",
            "Sensorimotor Systems",
        ],
        # --- Map canonical task codes -> domain ---
        "TASK2DOMAIN": {
            # Language & Regulatory Systems
            "emotionregulation": "Arousal & Regulatory Systems",
            # Cognitive Systems
            "audsem": "Cognitive Systems",
            "visrhyme": "Cognitive Systems",
            "vissem": "Cognitive Systems",
            "visspell": "Cognitive Systems",
            "arithmetic": "Cognitive Systems",
            "stroop": "Cognitive Systems",
            "cuedts": "Cognitive Systems",
            "axcpt": "Cognitive Systems",
            "matching": "Cognitive Systems",
            "stern": "Cognitive Systems",
            "st": "Cognitive Systems",
            "vswm": "Cognitive Systems",
            "expo": "Cognitive Systems",
            "recall": "Cognitive Systems",
            "feedback": "Cognitive Systems",
            "ppalocalizer": "Cognitive Systems",
            "localiser": "Cognitive Systems",
            "localizer": "Cognitive Systems",
            # Positive Valence System
            "fribbids": "Positive Valence System",
            "risk": "Positive Valence System",
            "itc": "Positive Valence System",
            # Negative Valence System
            "fearlearning": "Negative Valence System",
            "paingen": "Negative Valence System",
            # Sensorimotor
            "motor": "Sensorimotor Systems",
            "execution": "Sensorimotor Systems",
            "imagery": "Sensorimotor Systems",
            "ihg": "Sensorimotor Systems",
        },
    },
    "simulated": {
        # --- Categories of simulated task paradigms ---
        "DOMAIN_ORDER": [
            "Simulated Periodic",
            "Strong Performance on Real Data",
            "Weak Performance on Real Data",
        ],
        # --- Map task codes -> category ---
        "TASK2DOMAIN": {
            # Simulated Periodic
            "lowfreqlongrest": "Simulated Periodic",
            "lowfreqshortrest": "Simulated Periodic",
            "lowfreqshorttask": "Simulated Periodic",
            # Optimal Paradigm Design, Strong Performance on Real Data
            "axcpt": "Strong Performance on Real Data",
            "stern": "Strong Performance on Real Data",
            "cuedts": "Strong Performance on Real Data",
            "stroop": "Strong Performance on Real Data",
            # Optimal Paradigm Design, Weak Performance on Real Data
            "execution": "Weak Performance on Real Data",
            "imagery": "Weak Performance on Real Data",
            "localizer": "Weak Performance on Real Data",
            "ppalocalizer": "Weak Performance on Real Data",
            # Sub-Optimal Paradigm Design, Weak Performance on Real Data
            "itc": "Weak Performance on Real Data",
            "risk": "Weak Performance on Real Data",
        },
    },
}

###################### ml_results ######################


DEFAULT_EXPERIMENT_NAME_MAP = {
    "real": {
        "emotionregulation": "EXP.17",
        "audsem": "EXP.3",
        "visrhyme": "EXP.4",
        "vissem": "EXP.5",
        "visspell": "EXP.6",
        "arithmetic": "EXP.23",
        "stroop": "EXP.14",
        "cuedts": "EXP.12",
        "axcpt": "EXP.11",
        "matching": "EXP.24",
        "stern": "EXP.13",
        "st": "EXP.28",
        "vswm": "EXP.25",
        "expo": "EXP.19",
        "recall": "EXP.20",
        "feedback": "EXP.21",
        "ppalocalizer": "EXP.2",
        "localiser": "EXP.26",
        "localizer": "EXP.27",
        "fribbids": "EXP.10",
        "risk": "EXP.9",
        "itc": "EXP.8",
        "fearlearning": "EXP.1",
        "paingen": "EXP.22",
        "motor": "EXP.18",
        "execution": "EXP.15",
        "imagery": "EXP.16",
        "ihg": "EXP.7",
    },
    "simulated": {
        "lowfreqlongrest": "EXP.S.29",
        "lowfreqshortrest": "EXP.S.30",
        "lowfreqshorttask": "EXP.S.31",
        "axcpt": "EXP.S.11",
        "cuedts": "EXP.S.12",
        "stern": "EXP.S.13",
        "stroop": "EXP.S.14",
        "execution": "EXP.S.15",
        "imagery": "EXP.S.16",
        "localizer": "EXP.S.27",
        "ppalocalizer": "EXP.S.2",
        "itc": "EXP.S.8",
        "risk": "EXP.S.9",
    },
}


def canon_task(task_str: str) -> str:
    """strip 'task-' and non-letters, lowercase → canonical key"""
    s = task_str.replace("task-", "")
    s = re.sub(r"[^a-zA-Z]", "", s)
    return s.lower()


def get_default_experiment_name_map(simul_or_real: str):
    if simul_or_real not in DEFAULT_EXPERIMENT_NAME_MAP:
        raise ValueError(f"Invalid simul_or_real: {simul_or_real}")
    return DEFAULT_EXPERIMENT_NAME_MAP[simul_or_real].copy()


def get_present_task_order(tasks_iterable, task_reference_order):
    present_tasks = list(dict.fromkeys(tasks_iterable))
    present_set = set(present_tasks)
    ordered_tasks = [task for task in task_reference_order if task in present_set]
    remaining_tasks = sorted(
        [task for task in present_tasks if task not in ordered_tasks],
        key=lambda task: task.lower(),
    )
    return ordered_tasks + remaining_tasks


def _next_available_experiment_label(used_labels_lower):
    index = 1
    while f"exp{index}" in used_labels_lower:
        index += 1
    return f"exp{index}"


def build_experiment_display_info(tasks_iterable, task_reference_order, simul_or_real):
    """
    Resolve task order, experiment labels, and a stable palette for ML result plots.

    Edit ``DEFAULT_EXPERIMENT_NAME_MAP`` above to change experiment labels.
    Any task not listed there is auto-assigned the next available ``expN`` label.
    """
    task_order = get_present_task_order(tasks_iterable, task_reference_order)
    configured_map = get_default_experiment_name_map(simul_or_real)

    task_to_experiment = {}
    used_labels = {}
    used_labels_lower = set()

    for task in task_order:
        experiment_label = configured_map.get(canon_task(task))
        if experiment_label is None:
            experiment_label = _next_available_experiment_label(used_labels_lower)

        experiment_label_key = experiment_label.lower()
        if experiment_label_key in used_labels:
            raise ValueError(
                "Experiment labels must be unique for the plotted tasks. "
                f"Both '{used_labels[experiment_label_key]}' and '{task}' map to "
                f"'{experiment_label}'."
            )

        task_to_experiment[task] = experiment_label
        used_labels[experiment_label_key] = task
        used_labels_lower.add(experiment_label_key)

    n = max(1, len(task_order))
    colors = [
        _VIBRANT_DISTINCT_COLORS[i % len(_VIBRANT_DISTINCT_COLORS)] for i in range(n)
    ]
    experiment_order = [task_to_experiment[task] for task in task_order]
    experiment_palette = dict(zip(experiment_order, colors))

    return task_order, task_to_experiment, experiment_order, experiment_palette


def relabel_heatmap_rows(matrix_df, annot_df, task_reference_order, task_to_experiment):
    def _experiment_sort_key(exp_label):
        match = re.match(r"(?i)^\s*exp\s*[._-]?\s*(\d+)\s*$", str(exp_label))
        if match:
            return (0, int(match.group(1)), str(exp_label).lower())
        return (1, float("inf"), str(exp_label).lower())

    row_order = get_present_task_order(matrix_df.index.tolist(), task_reference_order)
    experiment_labels = [task_to_experiment[task] for task in row_order]

    relabeled_matrix = matrix_df.loc[row_order].copy()
    relabeled_matrix.index = experiment_labels

    relabeled_annot = None
    if annot_df is not None:
        relabeled_annot = annot_df.loc[row_order].copy()
        relabeled_annot.index = experiment_labels

    sorted_labels = sorted(relabeled_matrix.index.tolist(), key=_experiment_sort_key)
    relabeled_matrix = relabeled_matrix.loc[sorted_labels]
    if relabeled_annot is not None:
        relabeled_annot = relabeled_annot.loc[sorted_labels]

    return relabeled_matrix, relabeled_annot, row_order


def boldify_axes(ax, xlabel=None, ylabel=None, rotate_xticks=35):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")
    # dFC method names on x-axis
    if rotate_xticks is not None:
        plt.setp(
            ax.get_xticklabels(), fontweight="bold", rotation=rotate_xticks, ha="right"
        )
    else:
        plt.setp(ax.get_xticklabels(), fontweight="bold")


def mean_ci_boot(y, n_boot=3000, ci=95, rng=None):
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(y))
    if y.size == 1:
        return m, m, m
    if rng is None:
        rng = np.random.default_rng()  # fresh entropy
    idx = rng.integers(0, y.size, size=(n_boot, y.size))
    boots = np.mean(y[idx], axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return m, lo, hi


def summarize_methods_across_tasks(
    df_plot, ycol, method_col="dFC method", ci_func=mean_ci_boot
):
    """
    Return a DataFrame with columns: [method_col, 'mean','lo','hi'].
    Robust to Pandas quirks; no MultiIndex/unnamed columns.
    Assumes df_plot has one row per (task, method) already (your BEST table).
    """
    rows = []
    for meth, s in df_plot.groupby(method_col, observed=True)[ycol]:
        m, lo, hi = ci_func(s.values)
        rows.append({method_col: meth, "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def overlay_method_mean_ci(
    ax,
    df_plot,
    ycol,
    method_col="dFC method",
    line_halfwidth=0.30,
    cap_halfwidth=0.12,
    color="#222",
    lower=None,
    upper=None,
    rng=None,
):
    # map x positions from current ticks (call after you set/rotate xticklabels)
    xticks = ax.get_xticks()
    xlabs = [t.get_text() for t in ax.get_xticklabels()]
    xpos = {lab: xticks[i] for i, lab in enumerate(xlabs)}

    # summarize robustly
    summ = summarize_methods_across_tasks(
        df_plot, ycol, method_col, ci_func=lambda y: mean_ci_boot(y, rng=rng)
    )

    # clip to metric bounds if provided
    def clip(v):
        if lower is not None:
            v = max(lower, v)
        if upper is not None:
            v = min(upper, v)
        return v

    for _, r in summ.iterrows():
        meth = r[method_col]
        if meth not in xpos or np.isnan(r["mean"]):
            continue
        x = xpos[meth]
        m = clip(r["mean"])
        lo = clip(r["lo"]) if not np.isnan(r["lo"]) else m
        hi = clip(r["hi"]) if not np.isnan(r["hi"]) else m

        # mean line (thick) + CI whisker & caps (thin)
        ax.hlines(
            m, x - line_halfwidth, x + line_halfwidth, colors=color, lw=2.6, zorder=6
        )
        ax.vlines(x, lo, hi, colors=color, lw=1.2, alpha=0.9, zorder=5)
        ax.hlines(
            [lo, hi],
            x - cap_halfwidth,
            x + cap_halfwidth,
            colors=color,
            lw=1.2,
            alpha=0.9,
            zorder=5,
        )


###################### task_timing_stats ######################


def as_long_df(d, value_col, task_col="task"):
    rows = []
    for t, vals in d.items():
        for v in vals:
            rows.append({task_col: t, value_col: v})
    return pd.DataFrame(rows)


# --- median labels with matching hue colors (log-safe) ---
def annotate_medians_by_geometry(
    ax,
    df_long,
    x_col,
    hue_col,
    y_col,
    x_order,
    hue_order,
    fmt="{:.0f}",  # ints; change to "{:.2g}" if you prefer
    y_nudge_factor=1.08,
    bin_halfwidth=0.6,
    bbox_alpha=0.9,
):
    def _luminance(r, g, b):
        # simple relative luminance for contrast
        return 0.299 * r + 0.587 * g + 0.114 * b

    # collect box patches and centers
    patches = [
        p for p in getattr(ax, "artists", []) if isinstance(p, mpl.patches.PathPatch)
    ]
    if not patches:
        patches = [p for p in ax.patches if isinstance(p, mpl.patches.PathPatch)]

    boxes = []
    for p in patches:
        verts = p.get_path().vertices
        xs = verts[:, 0]
        x_center = 0.5 * (xs.min() + xs.max())
        boxes.append((x_center, p))

    if not boxes:
        return

    # bin by x tick index (0..len(x_order)-1)
    boxes_by_tick = {i: [] for i in range(len(x_order))}
    for x_center, p in boxes:
        idx = int(round(x_center))
        if idx in boxes_by_tick and abs(x_center - idx) <= bin_halfwidth:
            boxes_by_tick[idx].append((x_center, p))

    # medians from data
    med_dict = df_long.groupby([x_col, hue_col])[y_col].median().to_dict()

    for i, task in enumerate(x_order):
        group = boxes_by_tick.get(i, [])
        if not group:
            continue
        # left->right inside this task bin
        group.sort(key=lambda t: t[0])

        for j, hue in enumerate(hue_order):
            if j >= len(group):
                break
            x_center, patch = group[j]
            med = med_dict.get((task, hue), np.nan)
            if not (np.isfinite(med) and med > 0):
                continue

            # extract the exact facecolor of this box (matches legend/palette)
            fc = patch.get_facecolor()  # RGBA
            if fc is None or len(fc) < 3:
                # fallback (rare): use current color cycle
                fc = ax._get_lines.get_next_color()
                # normalize to RGBA
                fc = mpl.colors.to_rgba(fc)

            r, g, b, a = fc
            # adjust alpha for the textbox so it’s legible
            fc_box = (r, g, b, bbox_alpha)

            # choose black/white text for contrast
            txt_color = "black" if _luminance(r, g, b) > 0.6 else "white"

            ax.text(
                x_center,
                med * y_nudge_factor,
                fmt.format(med),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=txt_color,
                bbox=dict(boxstyle="round,pad=0.2", fc=fc_box, ec="none"),
                zorder=100,
                clip_on=False,
            )


# ---------- helpers: median ordering + median labeler (single-category boxplot) ----------
def order_by_median_dict(d, reverse=True):
    """Return (ordered_task_names, stats_dict) where stats_dict[task]=(median, std)."""
    stats = {t: (np.median(vals), np.std(vals)) for t, vals in d.items() if len(vals) > 0}
    ordered = sorted(stats.keys(), key=lambda t: stats[t][0], reverse=reverse)
    return ordered, stats


def annotate_medians_single_boxplot(
    ax, df_long, x_col, y_col, order, fmt="{:.2f}", box_alpha=0.90
):
    """
    Annotate the median for each category on a seaborn.boxplot *without hue*.
    Places the number at the geometric center of each box, using the box facecolor for the label bg.
    Call this AFTER setting any y-limits (so the nudge uses final limits).
    """

    # compute medians in plotting order
    med = df_long.groupby(x_col)[y_col].median().reindex(order)

    # collect PathPatches for boxes (artists in most seaborn versions; fallback to patches)
    patches = [
        p for p in getattr(ax, "artists", []) if isinstance(p, mpl.patches.PathPatch)
    ]
    if not patches:
        patches = [p for p in ax.patches if isinstance(p, mpl.patches.PathPatch)]

    n = min(len(patches), len(order))
    ymin, ymax = ax.get_ylim()
    dy = 0.02 * (ymax - ymin)  # small additive nudge in data units

    for k in range(n):
        patch = patches[k]
        verts = patch.get_path().vertices
        xs, _ = verts[:, 0], verts[:, 1]
        x_center = 0.5 * (xs.min() + xs.max())

        m = med.iloc[k]
        if not np.isfinite(m):
            continue

        # label background color = box facecolor (match legend/palette)
        fc = patch.get_facecolor()
        if fc is None or len(fc) < 3:
            fc = mpl.colors.to_rgba("white", box_alpha)
        else:
            fc = (fc[0], fc[1], fc[2], box_alpha)

        # text color for contrast (simple luminance check)
        lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
        txt_color = "black" if lum > 0.6 else "white"

        # keep label inside the axis (avoid hitting the top bound)
        y_text = min(m + dy, ymax - 0.01 * (ymax - ymin))

        ax.text(
            x_center,
            y_text,
            fmt.format(m),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=txt_color,
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec="none"),
            zorder=100,
            clip_on=False,
        )


###################### task_presence_binarization ######################

###################### dfc_visualization ######################


def _window_indices(
    trs, window_len=8, center="middle", center_time=None, center_index=None, interval=None
):
    T = len(trs)
    trs = np.asarray(trs)
    if interval is not None:
        t0, t1 = interval
        idxs = np.where((trs >= t0) & (trs <= t1))[0]
        if len(idxs) == 0:
            raise ValueError("interval produced no indices; check units.")
        return idxs
    if center_index is not None:
        c = int(np.clip(center_index, 0, T - 1))
    elif center_time is not None:
        c = int(np.argmin(np.abs(trs - center_time)))
    else:
        c = (T - 1) // 2
    half = window_len // 2
    start = max(0, c - half)
    end = min(T, start + window_len)
    start = max(0, end - window_len)
    return np.arange(start, end, dtype=int)


def _common_limits(dfc_dict, robust_percentile=(2, 98), symmetric=True):
    vals = []
    for A in dfc_dict.values():
        R = A.shape[1]
        iu = np.triu_indices(R, 1)
        vals.append(A[:, iu[0], iu[1]].ravel())
    lo, hi = np.percentile(np.concatenate(vals), robust_percentile)
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def figure_dfc_matrices_window_png(
    dfc_dict,
    trs,
    window_len=8,
    center="middle",
    center_time=None,
    center_index=None,
    interval=None,
    cmap="coolwarm",
    outfile="fig_dfc_window.png",
    show_region_ticks=False,
    region_labels=None,
    draw_network_bounds=None,
    dpi=600,
    transparent=False,
    # style knobs
    method_label_size=11,
    tr_label_size=10,
    cbar_label_size=11,
    rotate_method_labels=90,
    method_label_pad=18,  # << controls distance between method names and images
    wspace=None,  # << override column spacing if needed (None = auto)
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": tr_label_size,
            "axes.labelsize": method_label_size,
        }
    )

    methods = list(dfc_dict.keys())
    R = next(iter(dfc_dict.values())).shape[1]

    idxs = _window_indices(
        trs,
        window_len=window_len,
        center=center,
        center_time=center_time,
        center_index=center_index,
        interval=interval,
    )

    vmin, vmax = _common_limits(dfc_dict, robust_percentile=(2, 98), symmetric=True)
    vmin = 0

    # figure sizing
    col_width = 1.6
    row_height = 1.5
    nrows, ncols = len(methods), len(idxs)

    fig = plt.figure(figsize=((ncols + 0.5) * col_width, nrows * row_height))

    # spacing
    auto_wspace = min(0.35, 0.12 + 0.01 * ncols)
    wspace = auto_wspace if wspace is None else wspace
    hspace = 0.25

    # add a dedicated colorbar column on the far right
    gs = gridspec.GridSpec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.06],  # last slot = colorbar
        hspace=hspace,
        wspace=wspace,
    )

    last_im = None
    for r, m in enumerate(methods):
        A = dfc_dict[m]
        for c, t_idx in enumerate(idxs):
            ax = fig.add_subplot(gs[r, c])
            M = A[t_idx].copy()
            np.fill_diagonal(M, np.nan)
            im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")
            last_im = im

            if draw_network_bounds:
                for b in draw_network_bounds:
                    ax.axhline(b - 0.5, linewidth=0.4, color="k")
                    ax.axvline(b - 0.5, linewidth=0.4, color="k")

            if show_region_ticks and region_labels is not None:
                step = max(1, R // 16)
                ticks = np.arange(0, R, step)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(
                    [region_labels[i] for i in ticks], rotation=90, fontsize=6
                )
                ax.set_yticklabels([region_labels[i] for i in ticks], fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if r == 0:
                label = (
                    f"TR{trs[t_idx]}"
                    if np.issubdtype(np.asarray(trs).dtype, np.number)
                    else str(trs[t_idx])
                )
                ax.set_title(label, pad=6, fontsize=tr_label_size, fontweight="bold")

            if c == 0:
                ax.set_ylabel(
                    m,
                    rotation=rotate_method_labels,
                    labelpad=method_label_pad,  # << tighten/loosen here
                    va="center",
                    ha="center",
                    fontsize=method_label_size,
                    fontweight="bold",
                )
                ax.yaxis.set_label_position("left")

    # colorbar in its own axis (no overlap)
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Connectivity", fontsize=cbar_label_size, fontweight="bold")
    cbar.ax.tick_params(labelsize=max(8, cbar_label_size - 1))

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(
        outfile,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
        facecolor="white",
    )
    plt.close(fig)
    print(
        f"Saved {outfile}  |  TR columns: {len(idxs)}  |  vmin={vmin:.3f}, vmax={vmax:.3f}  |  dpi={dpi}"
    )


###################### sample_matrix plots ######################


def nice_step(n, max_ticks=10):
    """Return a 'nice' step (1-2-5x10^k) to keep ≤ max_ticks across [1..n]."""
    if n <= 1:
        return 1
    raw = max(1.0, n / max(2, (max_ticks - 1)))
    exp = np.floor(np.log10(raw))
    frac = raw / (10**exp)
    base = 1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10
    return int(base * (10**exp))


def plot_samples_features(
    X,
    y,
    *,
    sample_order="original",  # "original" | "label" | "label+cluster" | "cluster"
    feature_order="original",  # "original" | "tstat"
    col_order_from_train=None,  # optional np.ndarray (feature indices) to reuse on test
    ZSCORE=True,
    V_RANGE=None,
    cmap="coolwarm",
    title=None,
    save_path=None,
    show=True,
):
    """
    X: (n_samples, n_features) matrix (features in columns)
    y: (n_samples,) binary (0=rest, 1=task)

    Samples are shown along the horizontal axis (time-like), features along the vertical axis.
    If feature_order == "tstat", a slim vertical t-stat bar is shown on the LEFT,
    aligned 1:1 with feature rows (no top t-bar).
    """
    # ---------- prep ----------
    X = np.asarray(X, float)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    # z-score per feature
    Xz = X.copy()
    if ZSCORE:
        mu = Xz.mean(axis=0, keepdims=True)
        sd = Xz.std(axis=0, keepdims=True) + 1e-8
        Xz = (Xz - mu) / sd

    # ---------- feature order ----------
    if feature_order == "tstat":
        if col_order_from_train is not None:
            col_order = np.asarray(col_order_from_train, int)
            t, _ = ttest_ind(Xz[y == 1], Xz[y == 0], axis=0, equal_var=False)
            t_ord = t[col_order]
        else:
            t, _ = ttest_ind(Xz[y == 1], Xz[y == 0], axis=0, equal_var=False)
            col_order = np.argsort(-np.abs(t))  # strongest contrast first
            t_ord = t[col_order]
    else:
        col_order = np.arange(n_features)
        t_ord = None  # no t-stat bar

    # ---------- sample order ----------
    if sample_order == "original":
        row_order = np.arange(n_samples)
        split = np.sum(y == 0)
        draw_separator = False
    elif sample_order == "label":
        rest_idx = np.where(y == 0)[0]
        task_idx = np.where(y == 1)[0]
        row_order = np.r_[rest_idx, task_idx]
        split = len(rest_idx)
        draw_separator = True
    elif sample_order == "label+cluster":

        def order_rows(A):
            if len(A) <= 2:
                return np.arange(len(A))
            return leaves_list(linkage(A, method="average", metric="cosine"))

        rest_idx = np.where(y == 0)[0]
        task_idx = np.where(y == 1)[0]
        rest_order = rest_idx[order_rows(Xz[rest_idx])] if len(rest_idx) else rest_idx
        task_order = task_idx[order_rows(Xz[task_idx])] if len(task_idx) else task_idx
        row_order = np.r_[rest_order, task_order]
        split = len(rest_order)
        draw_separator = True
    elif sample_order == "cluster":

        def order_rows(A):
            if len(A) <= 2:
                return np.arange(len(A))
            return leaves_list(linkage(A, method="average", metric="cosine"))

        all_idx = np.arange(n_samples)
        # rest_order = rest_idx[order_rows(Xz[rest_idx])] if len(rest_idx) else rest_idx
        # task_order = task_idx[order_rows(Xz[task_idx])] if len(task_idx) else task_idx

        row_order = all_idx[order_rows(Xz[all_idx])]
        split = np.sum(y == 0)
        draw_separator = False
    else:
        raise ValueError(
            "sample_order must be one of {'original','label','label+cluster'}"
        )

    # ---------- figure & layout (no top t-bar) ----------
    # W = max(10, min(24, n_samples / 30))
    w_min = 12
    w_max = 24
    width_per_100 = 0.5  # additional width per 100 samples
    W = float(np.clip(w_min + (n_samples / 100.0) * width_per_100, w_min, w_max))
    H = max(6, min(16, n_features / 30))
    fig = plt.figure(figsize=(W, H))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 0.06],  # main heatmap + class strip
        hspace=0.08,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_lab = fig.add_subplot(gs[1, 0])

    # --- VRANGE ---
    if V_RANGE is None:
        Xflat = np.asarray(Xz, float).ravel()
        lo, hi = np.nanpercentile(Xflat, [5, 95])  # robust to outliers; tweak if needed
        V_RANGE = max(abs(lo), abs(hi))  # symmetric around 0 (for diverging cmap)

    # ---------- main heatmap ----------
    img = Xz[row_order, :][:, col_order].T  # (features, samples)
    ax_main.imshow(
        img, aspect="auto", origin="lower", cmap=cmap, vmin=-V_RANGE, vmax=V_RANGE
    )
    n_features = img.shape[0]
    last_idx = n_features - 1

    if n_features < 10:
        # every feature: labels 1..n, positions 0..n-1
        labels_1based = np.arange(1, n_features + 1, dtype=int)
    else:
        step = nice_step(n_features, max_ticks=10)
        # use round multiples of the step
        labels_1based = list(np.arange(step, n_features + 1, step, dtype=int))
        # de-dup & sort (in case step == 1)
        labels_1based = np.unique(labels_1based)

    # convert 1-based labels to 0-based tick positions
    ticks_pos = labels_1based - 1

    # lock y-limits so the last tick isn't clipped
    ax_main.set_ylim(-0.5, last_idx + 0.5)

    # set ticks & labels
    ax_main.set_yticks(ticks_pos)
    ax_main.set_yticklabels([f"{v:d}" for v in labels_1based])
    ax_main.set_ylabel("feature", fontsize=18, fontweight="bold")
    # ax_main.set_xlabel("sample", fontsize=18, fontweight="bold")
    # ax_main.set_xticks([])
    ax_main.tick_params(axis="y", labelsize=18)
    ax_main.tick_params(axis="x", labelsize=18)

    if draw_separator and 0 < split < n_samples:
        ax_main.axvline(split - 0.5, color="k", lw=2)

    # ---------- bottom class strip ----------
    y_reordered = y[row_order]
    cmap_lbl = ListedColormap(
        [[0.85, 0.85, 0.85], [0.25, 0.5, 0.9]]
    )  # rest=gray, task=blue
    ax_lab.imshow(
        y_reordered[None, :], aspect="auto", origin="lower", cmap=cmap_lbl, vmin=0, vmax=1
    )
    ax_lab.set_yticks([])
    ax_lab.set_xticks([])
    # ax_lab.set_title("class", fontsize=11, pad=2)

    # show class labels only when there is label grouping
    if draw_separator:
        n0 = (y_reordered == 0).sum()
        n1 = (y_reordered == 1).sum()
        if n0 > 0:
            x0 = (n0 - 1) / 2.0
            ax_lab.annotate(
                "rest (0)",
                xy=(x0, -0.35),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )
        if n1 > 0:
            x1 = n0 + (n1 - 1) / 2.0
            ax_lab.annotate(
                "task (1)",
                xy=(x1, -0.35),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )

    # --- move the class bar (ax_lab) down a bit ---
    fig.canvas.draw()  # ensure positions are current
    lab_box = ax_lab.get_position()  # [x0, y0, width, height] in figure coords
    down = 0.070  # how much to move down (figure fraction)
    new_y0 = max(0.01, lab_box.y0 - down)  # keep it inside the figure
    ax_lab.set_position([lab_box.x0, new_y0, lab_box.width, lab_box.height])

    # after you position ax_lab (i.e., after ax_lab.set_position([...]))
    ax_lab.xaxis.set_label_position("top")
    ax_lab.set_xlabel("sample", labelpad=4, fontweight="bold", fontsize=18)
    # keep the strip clean
    ax_lab.tick_params(
        axis="x", which="both", length=0, labelbottom=False, labeltop=False
    )

    # (re)grab the updated box for the colorbar placement that comes next
    lab_box = ax_lab.get_position()

    # ---------- LEFT vertical t-stat bar (only if feature_order=="tstat") ----------
    if t_ord is not None:
        fig.canvas.draw()
        main_box = ax_main.get_position()  # figure coords

        tbar_left_width = 0.010  # ~2% fig width
        tbar_left_pad = 0.035 / W * 24  # gap from main heatmap, proportional to fig width

        x0 = max(0.01, main_box.x0 - tbar_left_pad - tbar_left_width)
        y0 = main_box.y0
        w = tbar_left_width
        h = main_box.height

        ax_tleft = fig.add_axes([x0, y0, w, h])
        m = np.nanmax(np.abs(t_ord)) if np.isfinite(t_ord).any() else 1.0
        ax_tleft.imshow(
            t_ord[:, None], origin="lower", aspect="auto", cmap=cmap, vmin=-m, vmax=m
        )
        ax_tleft.set_xticks([])
        ax_tleft.set_yticks([])
        ax_tleft.set_title("t-stat", fontsize=11, pad=2, fontweight="bold")

    if title:
        fig.suptitle(title, y=0.995, fontsize=12, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(row_order=row_order, col_order=col_order)


def save_scalar_colorbar(
    cmap="coolwarm",
    vmin=-2.0,
    vmax=2.0,  # use the same V_RANGE you use in plots
    label="z-scored feature value",
    filename="zscore_colorbar.png",
    orientation="horizontal",
    figsize=(6, 0.4),  # width, height in inches
    dpi=300,
    ticks=None,
):
    """
    Saves a standalone scalar colorbar image you can reuse in the paper.
    """
    # Make a dummy mappable with the correct colormap and limits
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(
        [0.05, 0.35, 0.90, 0.30]
        if orientation == "horizontal"
        else [0.35, 0.05, 0.30, 0.90]
    )

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax, orientation=orientation)
    cb.set_label(label, fontsize=18, fontweight="bold")

    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.tick_params(labelsize=18)

    fig.savefig(filename, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
