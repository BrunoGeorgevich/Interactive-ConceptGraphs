import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import textwrap
import json
import os


def process_home_metrics(
    home_metrics_dirs: list[str],
    fraction_val: str,
    data: list,
    metrics_name_dict: dict,
    fraction_name_dict: dict,
    return_strs: bool = True,
) -> tuple:
    """
    Process home metrics directories and extract evaluation data.

    :param home_metrics_dirs: List of paths to home metric directories
    :type home_metrics_dirs: list[str]
    :param fraction_val: Fraction value as string
    :type fraction_val: str
    :param data: List to append metrics data to
    :type data: list
    :param metrics_name_dict: Dictionary mapping metric names to display names
    :type metrics_name_dict: dict
    :param fraction_name_dict: Dictionary mapping fraction values to display names
    :type fraction_name_dict: dict
    :param return_strs: Whether to return unique home and metric strings
    :type return_strs: bool
    :raises ValueError: If metrics JSON file count is invalid
    :raises FileNotFoundError: If metrics JSON file is not found
    :raises json.JSONDecodeError: If JSON parsing fails
    :return: Tuple containing data list and optionally home/metric strings
    :rtype: tuple
    """
    home_strs = []
    metric_strs = []

    for home_metric_dir in home_metrics_dirs:
        home_str = os.path.basename(os.path.dirname(home_metric_dir))
        metric_str = os.path.basename(home_metric_dir)

        home_strs.append(home_str)
        metric_strs.append(metric_str)

        metrics_json_path = glob(os.path.join(home_metric_dir, "metrics_*.json"))
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
        metrics_data["fraction"] = fraction_name_dict[fraction_val]
        data.append(metrics_data)

    if return_strs:
        home_strs = list(set(home_strs))
        metric_strs = list(set(metric_strs))
        return data, home_strs, metric_strs

    return data, None, None


def load_knowledge_sensitivity_data(
    data_path: str,
    zero_data_path: str,
    full_data_path: str,
) -> pd.DataFrame:
    """
    Load and combine knowledge sensitivity data from multiple directories.

    :param data_path: Path to the directory containing knowledge sensitivity CSV files
    :type data_path: str
    :param zero_data_path: Path to the directory containing zero-knowledge evaluation results
    :type zero_data_path: str
    :param full_data_path: Path to the directory containing full-knowledge evaluation results
    :type full_data_path: str
    :raises FileNotFoundError: If data directories are not found
    :raises ValueError: If data loading or processing fails
    :return: Combined DataFrame with all evaluation metrics
    :rtype: pd.DataFrame
    """
    try:
        fractions_dirs = glob(os.path.join(data_path, "*.*"))

        fractions_data = []
        home_strs = []
        metric_strs = []
        metrics_name_dict = {
            "adversarial_questions": "Graceful Failure",
            "basic_questions": "Direct",
            "follow_up_questions": "Follow-Up",
            "indirect_questions": "Indirect",
        }
        fraction_name_dict = {
            "0.0": "0",
            "0.25": "25",
            "0.50": "50",
            "0.75": "75",
            "1.0": "100",
        }

        for frac_dir in fractions_dirs:
            home_metrics_dirs = glob(os.path.join(frac_dir, "Home*", "*"))

            fraction_val = os.path.basename(frac_dir)
            fractions_data, home_strs_part, metric_strs_part = process_home_metrics(
                home_metrics_dirs,
                fraction_val,
                fractions_data,
                metrics_name_dict,
                fraction_name_dict,
                return_strs=True,
            )

            home_strs += home_strs_part
            metric_strs += metric_strs_part

        home_strs = list(set(home_strs))
        metric_strs = list(set(metric_strs))

        print(home_strs)
        print(metric_strs)

        zero_home_dirs = [
            os.path.join(zero_data_path, home, metric)
            for home in home_strs
            for metric in metric_strs
        ]

        fractions_data, _, _ = process_home_metrics(
            zero_home_dirs,
            "0.0",
            fractions_data,
            metrics_name_dict,
            fraction_name_dict,
            return_strs=False,
        )

        full_home_dirs = [
            os.path.join(full_data_path, home, metric)
            for home in home_strs
            for metric in metric_strs
        ]

        fractions_data, _, _ = process_home_metrics(
            full_home_dirs,
            "1.0",
            fractions_data,
            metrics_name_dict,
            fraction_name_dict,
            return_strs=False,
        )

        df = pd.DataFrame.from_dict(fractions_data)
        df["fraction"] = df["fraction"].astype(float)

        return df

    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to load knowledge sensitivity data: {e}") from e


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


def plot_sensitivity_lineplot(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot a line chart showing success rates across fractions for each metric.

    :param df: DataFrame containing knowledge sensitivity evaluation results
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
        plt.figure(figsize=(12, 6))

        grouped = (
            df.groupby(["fraction", "metric"])["success_rate"].mean().reset_index()
        )

        markers = {
            "Direct": "o",
            "Follow-Up": "s",
            "Indirect": "D",
            "Graceful Failure": "^",
        }

        palette = get_palette(attrs)
        ax = None
        for metric in grouped["metric"].unique():
            metric_data = grouped[grouped["metric"] == metric]
            if ax is None:
                ax = sns.lineplot(
                    data=metric_data,
                    x="fraction",
                    y="success_rate",
                    label=metric,
                    color=palette[list(grouped["metric"].unique()).index(metric)],
                    linewidth=3,
                    marker=markers[metric],
                    markersize=14,
                )
            else:
                sns.lineplot(
                    data=metric_data,
                    x="fraction",
                    y="success_rate",
                    label=metric,
                    color=palette[list(grouped["metric"].unique()).index(metric)],
                    linewidth=3,
                    marker=markers[metric],
                    markersize=14,
                    ax=ax,
                )

        metric_colors = {}
        for line, metric in zip(ax.get_lines(), grouped["metric"].unique()):
            metric_colors[metric] = line.get_color()

        # for metric in grouped["metric"].unique():
        #     metric_data = grouped[grouped["metric"] == metric]
        #     color = metric_colors[metric]

        # for _, row in metric_data.iterrows():
        #     ax.text(
        #         row["fraction"],
        #         row["success_rate"],
        #         f'{row["success_rate"]:.2f}',
        #         ha="center",
        #         va="bottom",
        #         fontsize=14,
        #         fontweight="bold",
        #         color="black",
        #         bbox=dict(
        #             boxstyle="round,pad=0.4",
        #             facecolor="white",
        #             edgecolor=color,
        #             linewidth=2,
        #         ),
        #     )

        ax.set_ylim(0.35, 1.05)
        ax.set_xlabel("Knowledge Fraction (%)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=16, fontweight="bold")
        ax.grid(axis="both", alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=14)

        ax.set_title(
            "Success Rate by Knowledge Fraction",
            fontweight="bold",
            pad=15,
            fontsize=18,
        )

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            frameon=True,
            ncol=4,
            fontsize=13,
            title="Question Type",
            title_fontsize=14,
        )

        save_plot(plt.gcf(), "knowledge_sensitivity_lineplot", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot sensitivity lineplot: {e}") from e


def plot_sensitivity_boxplot(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot boxplots showing success rate distribution across fractions.

    :param df: DataFrame containing knowledge sensitivity evaluation results
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

        plt.figure(figsize=(14, 6))

        df_all = df.copy()
        df_all["metric_type"] = "All Question Types"

        df_no_gf = df[df["metric"] != "Graceful Failure"].copy()
        df_no_gf["metric_type"] = "Without Graceful Failure"

        df_combined = pd.concat([df_all, df_no_gf], ignore_index=True)

        ax = sns.boxplot(
            data=df_combined,
            x="fraction",
            y="success_rate",
            hue="metric_type",
            palette=["#3498db", "#e74c3c"],
            showfliers=False,
        )

        np.random.seed(100)
        sns.stripplot(
            data=df_combined,
            x="fraction",
            y="success_rate",
            hue="metric_type",
            dodge=True,
            color="black",
            alpha=0.3,
            ax=ax,
            jitter=True,
            legend=False,
        )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Knowledge Fraction (%)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=16, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=14)

        ax.set_title(
            "Success Rate Distribution by Knowledge Fraction",
            fontweight="bold",
            pad=15,
            fontsize=18,
        )

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=True,
            ncol=2,
            fontsize=14,
        )

        save_plot(plt.gcf(), "knowledge_sensitivity_boxplot", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot sensitivity boxplot: {e}") from e


def plot_sensitivity_stacked_bar(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plot stacked bar chart showing response quality distribution across fractions.

    :param df: DataFrame containing knowledge sensitivity evaluation results
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

        grouped = df.groupby("fraction")[
            ["true_count", "partial_count", "false_count"]
        ].mean()

        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

        colors = ["#b6d7a8", "#ffe599", "#ea9999"]

        ax = grouped_pct[["true_count", "partial_count", "false_count"]].plot(
            kind="bar", stacked=True, color=colors, figsize=(12, 6), width=0.7
        )

        plt.title(
            "Response Quality by Knowledge Fraction", fontweight="bold", fontsize=20
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Percentage (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Knowledge Fraction (%)", fontsize=16, fontweight="bold")
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

        save_plot(plt.gcf(), "knowledge_sensitivity_stacked_bar", out_dir)

    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot sensitivity stacked bar: {e}") from e


def main() -> None:
    """
    Main function to load data and generate knowledge sensitivity plots.

    :raises FileNotFoundError: If input data files are not found
    :raises ValueError: If data loading or processing fails
    :return: None
    :rtype: None
    """
    DATABASE_PATH = r"d:\Documentos\Datasets\Robot@VirtualHomeLarge"
    DATA_DIR = os.path.join(
        DATABASE_PATH, "results", "ablation_study", "knowledge_sensitivity"
    )
    ZERO_DATA_DIR = os.path.join(DATABASE_PATH, "results", "interaction_eval_results")
    FULL_DATA_DIR = os.path.join(
        DATABASE_PATH, "results", "interaction_eval_results_with_ak"
    )
    OUTPUT_DIR = os.path.join(
        DATABASE_PATH, "results", "ablation_study", "plots_sensitivity"
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
        "sensitivity_lineplot": plot_sensitivity_lineplot,
        "sensitivity_boxplot": plot_sensitivity_boxplot,
        "sensitivity_stacked_bar": plot_sensitivity_stacked_bar,
    }

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading knowledge sensitivity data...")
        df = load_knowledge_sensitivity_data(DATA_DIR, ZERO_DATA_DIR, FULL_DATA_DIR)
        print(f"Data loaded: {len(df)} records.")

        print(f"Generating {len(PLOT_FUNCTIONS)} plots...")
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            try:
                print(f"  -> {plot_name}...")
                plot_func(df, plot_attributes, OUTPUT_DIR)
            except (KeyError, ValueError, TypeError) as e:
                traceback.print_exc()
                print(f"  [ERROR] Failed to generate '{plot_name}': {e}")

        print("All knowledge sensitivity plots generated successfully!")

    except (FileNotFoundError, ValueError) as e:
        traceback.print_exc()
        print(f"Critical error during execution: {e}")


if __name__ == "__main__":
    main()
