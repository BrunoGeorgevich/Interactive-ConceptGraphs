import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import textwrap
import json
import os


def load_short_term_memory_data(
    data_path: str,
) -> pd.DataFrame:
    """
    Load and combine short term memory data from multiple directories.

    :param data_path: Path to the directory containing short term memory CSV files
    :type data_path: str
    :raises FileNotFoundError: If data directories are not found
    :raises ValueError: If data loading or processing fails
    :return: Combined DataFrame with all evaluation metrics
    :rtype: pd.DataFrame
    """
    try:
        memories_dirs = glob(os.path.join(data_path, "with*"))

        memories_data = []
        metrics_name_dict = {
            "adversarial_questions": "Graceful Failure",
            "basic_questions": "Direct",
            "follow_up_questions": "Follow-Up",
            "indirect_questions": "Indirect",
            "temporal_consistency": "Temporal Consistency",
        }

        for mem_dir in memories_dirs:
            home_metrics_dirs = glob(os.path.join(mem_dir, "Home*", "*"))

            memory_val = os.path.basename(mem_dir)

            for home_metric_dir in home_metrics_dirs:
                home_str = os.path.basename(os.path.dirname(home_metric_dir))
                metric_str = os.path.basename(home_metric_dir)

                metrics_json_path = glob(
                    os.path.join(home_metric_dir, "metrics_*.json")
                )
                if len(metrics_json_path) == 0 or len(metrics_json_path) > 1:
                    raise ValueError(
                        f"Expected one metrics JSON file in {home_metric_dir}, found {len(metrics_json_path)}"
                    )

                metrics_json_path = metrics_json_path[0]
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

                metrics_data["home"] = home_str
                metrics_data["metric"] = (
                    metrics_name_dict[metric_str] if metrics_name_dict else metric_str
                )
                metrics_data["with_memory"] = memory_val == "with_short_term_memory"
                memories_data.append(metrics_data)

        df = pd.DataFrame.from_dict(memories_data)

        return df

    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to load short term memory data: {e}") from e


def apply_plot_style(plot_attributes: dict) -> None:
    """
    Apply global plotting styles using Matplotlib and Seaborn.

    :param plot_attributes: Configuration dictionary containing style settings
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

    :param plot_attributes: Configuration dictionary containing palette settings
    :type plot_attributes: dict
    :return: List or dictionary of colors for plotting
    :rtype: list | dict
    """
    return plot_attributes.get("palette", sns.color_palette("viridis", 5))


def wrap_label(label: str, width: int = 14) -> str:
    """
    Wrap label text to the specified width for improved readability.

    :param label: The label string to wrap
    :type label: str
    :param width: Maximum line width before wrapping
    :type width: int
    :return: Wrapped label string with newlines
    :rtype: str
    """
    return "\n".join(textwrap.wrap(label, width=width))


def save_plot(fig: plt.Figure, name: str, output_dir: str) -> None:
    """
    Save the plot to disk and close the figure.

    :param fig: Matplotlib figure object to save
    :type fig: plt.Figure
    :param name: Base name for the output file
    :type name: str
    :param output_dir: Directory path where the plot will be saved
    :type output_dir: str
    :raises OSError: If the file cannot be saved to the specified path
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


def plot_memories_boxplot(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot boxplots showing success rate distribution with and without memory for each metric.

    :param df: DataFrame containing short term memory evaluation results
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes
    :type attrs: dict
    :param out_dir: Directory to save the generated plot
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame
    :raises ValueError: If data processing fails
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)

        df_plot = df.copy()
        df_plot["memory_status"] = df_plot["with_memory"].map(
            {True: "With Memory", False: "Without Memory"}
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        sns.boxplot(
            data=df_plot,
            x="metric",
            y="success_rate",
            hue="memory_status",
            palette=["#3498db", "#e74c3c"],
            showfliers=False,
            ax=ax,
        )

        np.random.seed(100)
        sns.stripplot(
            data=df_plot,
            x="metric",
            y="success_rate",
            hue="memory_status",
            dodge=True,
            color="black",
            alpha=0.3,
            ax=ax,
            jitter=True,
            legend=False,
        )

        ax.set_ylim(0.2, 1.05)
        ax.set_xlabel("Metric Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=16, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xticklabels(
            [wrap_label(label.get_text(), width=14) for label in ax.get_xticklabels()],
            rotation=0,
        )

        ax.set_title(
            "Success Rate by Metric Type: With and Without Short-term Memory",
            fontweight="bold",
            pad=15,
            fontsize=18,
        )

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),
            frameon=True,
            ncol=2,
            fontsize=14,
        )

        save_plot(fig, "short_term_memory_boxplot", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot memories boxplot: {e}") from e


def plot_memories_stacked_bar(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot stacked bar chart showing response quality distribution across memories.

    :param df: DataFrame containing short term memory evaluation results
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes
    :type attrs: dict
    :param out_dir: Directory to save the generated plot
    :type out_dir: str
    :raises KeyError: If required columns are missing from the DataFrame
    :raises ValueError: If data processing fails
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)

        grouped = df.groupby("memory")[
            ["true_count", "partial_count", "false_count"]
        ].mean()

        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

        colors = ["#b6d7a8", "#ffe599", "#ea9999"]

        ax = grouped_pct[["true_count", "partial_count", "false_count"]].plot(
            kind="bar", stacked=True, color=colors, figsize=(12, 6), width=0.7
        )

        plt.title(
            "Response Quality by Knowledge memory", fontweight="bold", fontsize=20
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Percentage (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Knowledge memory (%)", fontsize=16, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticklabels(
            [f"{float(lbl.get_text()):.0f}" for lbl in ax.get_xticklabels()],
            fontsize=14,
            rotation=0,
        )
        ax.tick_params(axis="y", labelsize=14)

        plt.legend(
            ["True", "Partial", "False"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=True,
            ncol=3,
            fontsize=14,
        )

        for c in ax.containers:
            labels = [
                f"{v.get_height():.1f}%" if v.get_height() >= 3.0 else "" for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                color="black",
                fontweight="bold",
                fontsize=12,
            )

        save_plot(plt.gcf(), "short_term_memory_stacked_bar", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot memories stacked bar: {e}") from e


def main() -> None:
    """
    Main function to load data and generate short term memory plots.

    :raises FileNotFoundError: If input data files are not found
    :raises ValueError: If data loading or processing fails
    :return: None
    :rtype: None
    """
    DATABASE_PATH = r"d:\Documentos\Datasets\Robot@VirtualHomeLarge"
    DATA_DIR = os.path.join(
        DATABASE_PATH, "results", "ablation_study", "short_term_memory_impact"
    )
    OUTPUT_DIR = os.path.join(
        DATABASE_PATH, "results", "ablation_study", "plots_memories"
    )

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
        "memories_boxplot": plot_memories_boxplot,
        # "memories_stacked_bar": plot_memories_stacked_bar,
    }

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading short term memory data...")
        df = load_short_term_memory_data(DATA_DIR)
        print(f"Data loaded: {len(df)} records.")

        print(f"Generating {len(PLOT_FUNCTIONS)} plots...")
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            try:
                print(f"  -> {plot_name}...")
                plot_func(df, plot_attributes, OUTPUT_DIR)
            except (KeyError, ValueError, TypeError) as e:
                traceback.print_exc()
                print(f"  [ERROR] Failed to generate '{plot_name}': {e}")

        print("All short term memory plots generated successfully!")

    except (FileNotFoundError, ValueError) as e:
        traceback.print_exc()
        print(f"Critical error during execution: {e}")


if __name__ == "__main__":
    main()
