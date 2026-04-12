#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper_functions import (
    annotate_medians_single_boxplot,
    build_experiment_display_info,
    canon_task,
    get_default_experiment_name_map,
    order_by_median_dict,
    setup_pub_style,
)


def _load_and_filter_tsnr_df(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    # Make sure expected columns exist
    required_cols = {"dataset", "sub", "ses", "task", "run", "tsnr_median", "error"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in TSV: {sorted(missing)}")

    # Normalize missing values
    df = df.fillna("")

    # Keep only rows without errors
    df = df[df["error"].astype(str).str.strip() == ""].copy()

    # Keep only the desired session for multi-session datasets
    # ds005038 -> keep ses == "pre"
    # ds003823 -> keep ses == "post"
    mask_ds005038 = df["dataset"] == "ds005038"
    mask_ds003823 = df["dataset"] == "ds003823"

    df = df[
        (~mask_ds005038 | (df["ses"] == "pre")) & (~mask_ds003823 | (df["ses"] == "post"))
    ].copy()

    # Convert tSNR median to numeric
    df["tsnr_median"] = pd.to_numeric(df["tsnr_median"], errors="coerce")

    # Drop rows where tsnr_median could not be parsed
    df = df[df["tsnr_median"].notna()].copy()

    return df


def build_grouped_tsnr_summary(tsv_path: str) -> Path:
    tsv_path = Path(tsv_path).resolve()
    out_path = tsv_path.parent / "tsnr_summary_grouped.tsv"

    df = _load_and_filter_tsnr_df(str(tsv_path))

    # Average over subjects for each dataset/task/run
    out_df = (
        df.groupby(["dataset", "task", "run"], as_index=False)["tsnr_median"]
        .mean()
        .rename(columns={"tsnr_median": "median_tsnr_avg_over_subjects"})
    )

    # Append prefixes
    out_df["task"] = "task-" + out_df["task"].astype(str)

    def format_run(x):
        if pd.isna(x) or str(x).strip() == "":
            return None
        return f"run-{x}"

    out_df["run"] = out_df["run"].apply(format_run)

    # Reorder columns exactly as requested
    out_df = out_df[["dataset", "run", "task", "median_tsnr_avg_over_subjects"]]

    # Optional: round nicely
    out_df["median_tsnr_avg_over_subjects"] = out_df[
        "median_tsnr_avg_over_subjects"
    ].round(2)

    # Save in same directory
    out_df.to_csv(out_path, sep="\t", index=False)

    return out_path


def build_tsnr_distribution_figure(tsv_path: str) -> Path:
    tsv_path = Path(tsv_path).resolve()
    fig_path = tsv_path.parent / "tsnr_median_distribution_by_exp.png"

    df = _load_and_filter_tsnr_df(str(tsv_path))
    if df.empty:
        raise ValueError("No valid tSNR rows available to plot after filtering.")

    task_to_values = df.groupby("task")["tsnr_median"].apply(list).to_dict()
    if not task_to_values:
        raise ValueError("No task-wise tSNR values found for plotting.")

    tasks_present = sorted(task_to_values.keys())
    known_tasks = set(get_default_experiment_name_map("real").keys())
    unknown_tasks = sorted(
        [task for task in tasks_present if canon_task(task) not in known_tasks]
    )
    if unknown_tasks:
        unknown_str = ", ".join(unknown_tasks)
        raise ValueError(
            "Found task(s) not mapped to EXP labels in real-data mapping: "
            f"{unknown_str}. Remove these tasks from input TSV or add them to "
            "DEFAULT_EXPERIMENT_NAME_MAP['real'] in helper_functions.py."
        )

    _, task_to_experiment, _, _ = build_experiment_display_info(
        tasks_iterable=tasks_present,
        task_reference_order=tasks_present,
        simul_or_real="real",
    )

    order_task, _ = order_by_median_dict(task_to_values, reverse=True)
    order_exp = [task_to_experiment[t] for t in order_task]

    df_plot = df.copy()
    df_plot["experiment"] = df_plot["task"].map(task_to_experiment)
    df_plot = df_plot[df_plot["task"].isin(order_task)].copy()
    df_plot["experiment"] = pd.Categorical(
        df_plot["experiment"], categories=order_exp, ordered=True
    )

    setup_pub_style()
    sns.set_theme(context="paper", style="darkgrid")

    fig_w = max(14, 14 / 30 * len(order_exp))
    plt.figure(figsize=(fig_w, 6))
    ax = sns.boxplot(
        data=df_plot,
        x="experiment",
        y="tsnr_median",
        order=order_exp,
        width=0.6,
        linewidth=1,
        showfliers=False,
    )

    annotate_medians_single_boxplot(
        ax,
        df_plot,
        x_col="experiment",
        y_col="tsnr_median",
        order=order_exp,
        fmt="{:.1f}",
        box_alpha=0.6,
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("tSNR median")
    for label in ax.get_xticklabels():
        label.set_rotation(65)
        label.set_horizontalalignment("right")
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.1, dpi=500)
    plt.close()

    return fig_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a grouped TSV from tsnr_summary.tsv and create a figure showing "
            "tSNR median distributions per experiment (EXP)."
        )
    )
    parser.add_argument(
        "tsnr_summary_tsv",
        help="Path to tsnr_summary.tsv",
    )
    args = parser.parse_args()

    out_path = build_grouped_tsnr_summary(args.tsnr_summary_tsv)
    fig_path = build_tsnr_distribution_figure(args.tsnr_summary_tsv)
    print(f"[DONE] Wrote grouped TSV to: {out_path}")
    print(f"[DONE] Wrote figure to: {fig_path}")


if __name__ == "__main__":
    main()
