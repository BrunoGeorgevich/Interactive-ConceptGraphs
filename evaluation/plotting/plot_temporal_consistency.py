import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import textwrap
import os

"""
Module for plotting temporal consistency evaluation results.

This module provides functions to generate various plots for analyzing
temporal consistency evaluation data across different strategies.
"""

import matplotlib.pyplot as plt


def load_temporal_consistency_data(
    path_clip: str,
    path_llm: str,
    path_system: str,
    path_system_without_ak: str,
    path_system_with_ak: str,
) -> pd.DataFrame:
    """
    Load and combine temporal consistency CSV files into a single DataFrame.

    :param path_clip: Path to the ConceptGraphs (CLIP) CSV file.
    :type path_clip: str
    :param path_llm: Path to the ConceptGraphs (LLM) CSV file.
    :type path_llm: str
    :param path_system: Path to the HIPaMS without Learning CSV file.
    :type path_system: str
    :param path_system_without_ak: Path to the HIPaMS with Learning CSV file.
    :type path_system_without_ak: str
    :param path_system_with_ak: Path to the HIPaMS with Learning + AK CSV file.
    :type path_system_with_ak: str
    :raises FileNotFoundError: If any of the specified files do not exist.
    :raises pd.errors.ParserError: If CSV parsing fails.
    :return: Combined DataFrame with a 'Strategy' column identifying each source.
    :rtype: pd.DataFrame
    """
    try:
        df_clip = pd.read_csv(path_clip, delimiter=",")
        df_llm = pd.read_csv(path_llm, delimiter=",")
        df_system = pd.read_csv(path_system, delimiter=",")
        df_system_without_ak = pd.read_csv(path_system_without_ak, delimiter=",")
        df_system_with_ak = pd.read_csv(path_system_with_ak, delimiter=",")

        df_clip["Strategy"] = "ConceptGraphs (CLIP)"
        df_llm["Strategy"] = "ConceptGraphs (LLM)"
        df_system["Strategy"] = "HIPaMS without Learning"
        df_system_without_ak["Strategy"] = "HIPaMS"
        df_system_with_ak["Strategy"] = "HIPaMS + Knowledge"

        df_combined = pd.concat(
            [df_clip, df_llm, df_system, df_system_without_ak, df_system_with_ak],
            ignore_index=True,
        )

        return df_combined

    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to load temporal consistency data: {e}") from e


def apply_plot_style(plot_attributes: dict) -> None:
    """
    Apply global plotting styles using Matplotlib and Seaborn.

    :param plot_attributes: Configuration dictionary containing style settings.
    :type plot_attributes: dict
    :return: None
    :rtype: None
    """
    sns.set_theme(style=plot_attributes.get("style", "whitegrid"))
    sns.set_context("notebook", font_scale=plot_attributes.get("font_scale", 1.2))
    plt.rcParams["font.family"] = plot_attributes.get("font_family", "sans-serif")
    plt.rcParams["figure.figsize"] = plot_attributes.get("figsize", (12, 8))
    plt.rcParams["axes.titlesize"] = plot_attributes.get("title_size", 18)
    plt.rcParams["axes.labelsize"] = plot_attributes.get("label_size", 14)
    plt.rcParams["savefig.dpi"] = plot_attributes.get("dpi", 300)


def get_palette(plot_attributes: dict) -> list | dict:
    """
    Retrieve the color palette from attributes.

    :param plot_attributes: Configuration dictionary containing palette settings.
    :type plot_attributes: dict
    :return: List or dictionary of colors for plotting.
    :rtype: list | dict
    """
    return plot_attributes.get("palette", sns.color_palette("viridis", 5))


def wrap_label(label: str, width: int = 14) -> str:
    """
    Wrap label text to the specified width for improved readability.

    :param label: The label string to wrap.
    :type label: str
    :param width: Maximum line width before wrapping.
    :type width: int
    :return: Wrapped label string with newlines.
    :rtype: str
    """
    return "\n".join(textwrap.wrap(label, width=width))


def save_plot(fig: plt.Figure, name: str, output_dir: str) -> None:
    """
    Save the plot to disk and close the figure.

    :param fig: Matplotlib figure object to save.
    :type fig: plt.Figure
    :param name: Base name for the output file.
    :type name: str
    :param output_dir: Directory path where the plot will be saved.
    :type output_dir: str
    :raises OSError: If the file cannot be saved to the specified path.
    :return: None
    :rtype: None
    """
    try:
        path = os.path.join(output_dir, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except (OSError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save plot to {output_dir}: {e}") from e


def plot_temporal_success_bar(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot a grouped bar chart comparing temporal consistency success rates.

    :param df: DataFrame containing temporal consistency evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the generated plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            data=df,
            x="Strategy",
            y="success_rate",
            hue="Strategy",
            palette=get_palette(attrs),
            errorbar="sd",
            capsize=0.2,
        )

        wrapped_labels = [
            wrap_label(lbl.get_text(), 12) for lbl in ax.get_xticklabels()
        ]
        ax.set_xticklabels(wrapped_labels, fontsize=14)

        ax.set_title(
            "Temporal Consistency Success Rate", fontweight="bold", pad=15, fontsize=18
        )
        ax.set_ylabel("Success Rate", fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", alpha=0.5)

        save_plot(plt.gcf(), "temporal_consistency_success_bar", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot temporal success bar: {e}") from e


def plot_temporal_boxplot(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot a boxplot showing distribution of temporal consistency success rates.

    :param df: DataFrame containing temporal consistency evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the generated plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(8, 5))

        ax = sns.boxplot(
            data=df,
            x="Strategy",
            y="success_rate",
            hue="Strategy",
            palette=get_palette(attrs),
            showfliers=False,
        )

        np.random.seed(100)
        sns.stripplot(
            data=df,
            x="Strategy",
            y="success_rate",
            color="black",
            alpha=0.5,
            ax=ax,
            jitter=True,
        )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_title(
            "Distribution of Success Rates (Temporal Consistency)",
            fontweight="bold",
            pad=10,
            fontsize=16,
        )
        ax.set_ylabel("", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)

        handles, labels = ax.get_legend_handles_labels()
        if not handles or not labels:
            unique_strategies = df["Strategy"].unique()
            handles = [
                plt.Line2D([0], [0], color=color, lw=4) for color in get_palette(attrs)
            ]
            labels = [str(s) for s in unique_strategies]

        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.46, -0.35),
            frameon=True,
            ncol=2,
            fontsize=15,
        )

        save_plot(plt.gcf(), "temporal_consistency_boxplot", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot temporal boxplot: {e}") from e


def plot_temporal_stacked_bar(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot a stacked bar chart showing response quality distribution.

    :param df: DataFrame containing temporal consistency evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the generated plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)

        grouped = df.groupby("Strategy")[
            ["true_count", "partial_count", "false_count"]
        ].mean()

        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

        grouped_pct = grouped_pct.reindex(
            [
                "ConceptGraphs (CLIP)",
                "ConceptGraphs (LLM)",
                "HIPaMS without Learning",
                "HIPaMS",
                "HIPaMS + Knowledge",
            ]
        )

        colors = ["#b6d7a8", "#ffe599", "#ea9999"]

        ax = grouped_pct[["true_count", "partial_count", "false_count"]].plot(
            kind="bar", stacked=True, color=colors, figsize=(12, 5), width=0.7
        )

        plt.title(
            "Temporal Consistency Response Quality", fontweight="bold", fontsize=24
        )
        ax.grid(False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.set_xticklabels(
            [wrap_label(lbl.get_text(), 12) for lbl in ax.get_xticklabels()],
            fontsize=19,
            rotation=0,
        )

        plt.legend(
            ["True", "Partial", "False"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.45),
            frameon=True,
            ncol=3,
            fontsize=20,
        )

        for c in ax.containers:
            labels = [
                f"{v.get_height():.1f}%" if v.get_height() >= 1.0 else "" for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                color="black",
                fontweight="bold",
                fontsize=18,
            )

        save_plot(plt.gcf(), "temporal_consistency_stacked_bar", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot temporal stacked bar: {e}") from e


def plot_temporal_heatmap(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot a heatmap of temporal consistency success rates per home and strategy.

    :param df: DataFrame containing temporal consistency evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the generated plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(12, 14))

        pivot = df.pivot(index="home_id", columns="Strategy", values="success_rate")

        wrapped_columns = [wrap_label(col, 12) for col in pivot.columns]

        ax = sns.heatmap(
            pivot,
            annot=True,
            cmap="Purples",
            fmt=".2f",
            cbar_kws={"ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            annot_kws={"size": 14},
            vmin=0.0,
            vmax=1.0,
        )

        if ax.figure.axes and len(ax.figure.axes) > 1:
            colorbar = ax.figure.axes[-1]
            colorbar.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            colorbar.set_ylim(0.0, 1.0)
            colorbar.tick_params(labelsize=14)

        ax.set_xticklabels(wrapped_columns, rotation=0, fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        ax.set_title(
            "Temporal Consistency per Home", fontweight="bold", fontsize=20, pad=20
        )
        ax.set_ylabel("Home ID", fontsize=20, fontweight="bold")
        ax.set_xlabel("")

        save_plot(plt.gcf(), "temporal_consistency_heatmap", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot temporal heatmap: {e}") from e


def main() -> None:
    """
    Main function to load data and generate temporal consistency plots.

    :raises FileNotFoundError: If input data files are not found.
    :raises ValueError: If data loading or processing fails.
    :return: None
    :rtype: None
    """
    DATABASE_PATH = r"d:\Documentos\Datasets\Robot@VirtualHomeLarge"
    DATA_DIR = os.path.join(
        DATABASE_PATH, "results", "interaction_eval_temporal_consistency"
    )
    OUTPUT_DIR = os.path.join(
        DATABASE_PATH, "results", "evaluation_results", "plots_temporal"
    )

    PATH_CLIP = os.path.join(DATA_DIR, "summary_clip.csv")
    PATH_LLM = os.path.join(DATA_DIR, "summary_llm.csv")
    PATH_SYSTEM = os.path.join(DATA_DIR, "summary_system.csv")
    PATH_SYSTEM_WITHOUT_AK = os.path.join(DATA_DIR, "summary_system_without_ak.csv")
    PATH_SYSTEM_WITH_AK = os.path.join(DATA_DIR, "summary_system_with_ak.csv")

    plot_attributes = {
        "style": "whitegrid",
        "font_family": "sans-serif",
        "font_scale": 1.1,
        "figsize": (12, 7),
        "title_size": 18,
        "label_size": 14,
        "dpi": 300,
        "palette": [
            "#ef563c",
            "#f5b700",
            "#3498db",
            "#07ca98",
            "#6671fa",
        ],
    }

    PLOT_FUNCTIONS = {
        # "temporal_success_bar": plot_temporal_success_bar,
        "temporal_boxplot": plot_temporal_boxplot,
        "temporal_stacked_bar": plot_temporal_stacked_bar,
        # "temporal_heatmap": plot_temporal_heatmap,
    }

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading temporal consistency data...")
        df = load_temporal_consistency_data(
            PATH_CLIP,
            PATH_LLM,
            PATH_SYSTEM,
            PATH_SYSTEM_WITHOUT_AK,
            PATH_SYSTEM_WITH_AK,
        )
        print(f"Data loaded: {len(df)} records.")

        print(f"Generating {len(PLOT_FUNCTIONS)} plots...")
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            try:
                print(f"  -> {plot_name}...")
                plot_func(df, plot_attributes, OUTPUT_DIR)
            except (KeyError, ValueError, TypeError) as e:
                traceback.print_exc()
                print(f"  [ERROR] Failed to generate '{plot_name}': {e}")

        print("All temporal consistency plots generated successfully!")

    except (FileNotFoundError, ValueError) as e:
        traceback.print_exc()
        print(f"Critical error during execution: {e}")


if __name__ == "__main__":
    main()
