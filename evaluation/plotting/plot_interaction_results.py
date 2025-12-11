import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import textwrap
import math
import os


def load_and_preprocess_data(
    path_ak: str, path_improved: str, path_original: str
) -> pd.DataFrame:
    """
    Loads data from the three specified CSV files, normalizes the format,
    and combines them into a single DataFrame for analysis.

    :param path_ak: Path to the Improved + Additional Knowledge CSV.
    :type path_ak: str
    :param path_improved: Path to the Improved CSV.
    :type path_improved: str
    :param path_original: Path to the Original CSV.
    :type path_original: str
    :return: A combined DataFrame with a 'Strategy' column.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If any of the files are not found.
    :raises ValueError: If data loading fails.
    """
    try:
        df_ak = pd.read_csv(path_ak, delimiter=";")
        df_imp = pd.read_csv(path_improved, delimiter=";")
        df_orig = pd.read_csv(path_original, delimiter=",")

        df_ak["Strategy"] = "Ours + Knowledge"
        df_imp["Strategy"] = "Ours"
        df_orig["Strategy"] = "ConceptGraphs"

        df_combined = pd.concat([df_orig, df_imp, df_ak], ignore_index=True)

        return df_combined
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error loading data: {e}")


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the wide DataFrame into a long format suitable for Seaborn plotting.
    Correctly handles composite prefixes like 'follow_up'.

    :param df: The combined wide-format DataFrame.
    :type df: pd.DataFrame
    :return: Long-format DataFrame with columns ['home_id', 'Strategy', 'Question Type', 'Metric', 'Value'].
    :rtype: pd.DataFrame
    """

    metrics_map = {
        "success_rate": "Success Rate",
        "true": "True Count",
        "partial": "Partial Count",
        "false": "False Count",
        "total": "Total Questions",
    }

    types_map = {
        "adversarial": "Graceful Failure",
        "basic": "Basic",
        "follow_up": "Follow-up",
        "indirect": "Indirect",
        "overall": "Overall",
    }

    melted_rows = []

    for _, row in df.iterrows():
        for col in df.columns:
            if col in ["home_id", "Strategy"]:
                continue

            parts = col.split("_")

            q_type = None
            metric_start_idx = 0

            if len(parts) > 1:
                composite_key = f"{parts[0]}_{parts[1]}"
                if composite_key in types_map:
                    q_type = types_map[composite_key]
                    metric_start_idx = 2

            if q_type is None and parts[0] in types_map:
                q_type = types_map[parts[0]]
                metric_start_idx = 1

            if not q_type:
                continue

            metric_part = "_".join(parts[metric_start_idx:])
            metric_key = metric_part.replace("questions_", "")

            if metric_key in metrics_map:
                metric_name = metrics_map[metric_key]
                melted_rows.append(
                    {
                        "home_id": row["home_id"],
                        "Strategy": row["Strategy"],
                        "Question Type": q_type,
                        "Metric": metric_name,
                        "Value": row[col],
                    }
                )

    return pd.DataFrame(melted_rows)


def apply_plot_style(plot_attributes: dict) -> None:
    """
    Applies global plotting styles using Matplotlib and Seaborn.

    :param plot_attributes: Configuration dictionary.
    :type plot_attributes: dict
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
    Retrieves the color palette from attributes.

    :param plot_attributes: Configuration dictionary.
    :type plot_attributes: dict
    :return: List or Dictionary of colors.
    :rtype: list | dict
    """
    return plot_attributes.get("palette", sns.color_palette("viridis", 3))


def save_plot(fig: plt.Figure, name: str, output_dir: str) -> None:
    """Helper to save the plot and close the figure."""
    # path = os.path.join(output_dir, f"{name}.png")
    path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def wrap_label(label: str, width: int = 14) -> str:
    """
    Wraps the label text to the specified width.

    :param label: The label string to wrap.
    :type label: str
    :param width: Maximum line width before wrapping.
    :type width: int
    :return: Wrapped label string.
    :rtype: str
    """
    return "\n".join(textwrap.wrap(label, width=width))


def plot_01_overall_success_bar(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Plots a bar chart comparing the overall success rate (mean + CI) for each strategy,
    wrapping x-axis labels for improved readability.

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(6, 6))

        data = df[(df["Question Type"] == "Overall") & (df["Metric"] == "Success Rate")]

        ax = sns.barplot(
            data=data,
            x="Strategy",
            y="Value",
            hue="Strategy",
            palette=get_palette(attrs),
            errorbar="sd",
            capsize=0.2,
        )

        wrapped_labels = [wrap_label(lbl.get_text()) for lbl in ax.get_xticklabels()]
        ax.set_xticklabels(wrapped_labels, fontsize=15)

        ax.set_title("Overall Success Rate Comparison", fontweight="bold", pad=15)
        ax.set_ylabel("Success Rate", fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        save_plot(plt.gcf(), "01_overall_success_bar", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot overall success bar: {e}")


def plot_02_type_success_grouped_bar(
    df: pd.DataFrame, attrs: dict, out_dir: str
) -> None:
    """
    Plots a grouped bar chart of success rate by question type for each strategy,
    with the legend displayed in a box below the chart and centered.

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(8, 8))

        data = df[(df["Question Type"] != "Overall") & (df["Metric"] == "Success Rate")]

        ax = sns.barplot(
            data=data,
            x="Question Type",
            y="Value",
            hue="Strategy",
            palette=get_palette(attrs),
            errorbar="sd",
            capsize=0.2,
        )

        ax.set_title(
            "Success Rate by Question Type",
            fontweight="bold",
            pad=25,
            fontdict={"fontsize": 24},
        )
        ax.set_ylabel("Success Rate", fontsize=20, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()], fontsize=16)
        ax.set_xticklabels(
            [wrap_label(lbl.get_text(), width=14) for lbl in ax.get_xticklabels()],
            fontsize=16,
        )
        ax.set_xlabel("Question Type", fontsize=18, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.46, -0.25),
            frameon=True,
            ncol=len(data["Strategy"].unique()),
            fontsize=16,
        )

        save_plot(plt.gcf(), "02_type_success_grouped_bar", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot type success grouped bar: {e}")


def plot_03_response_quality_stacked(
    df: pd.DataFrame, attrs: dict, out_dir: str
) -> None:
    """
    Plots a stacked bar chart showing the distribution of True, Partial, and False answers (Overall),
    ensuring the 'Original' strategy is displayed on the left.

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)

        metrics = ["True Count", "Partial Count", "False Count"]
        data = df[(df["Question Type"] == "Overall") & (df["Metric"].isin(metrics))]

        grouped = data.groupby(["Strategy", "Metric"])["Value"].mean().unstack()

        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

        strategy_order = ["ConceptGraphs"] + [
            s for s in grouped_pct.index if s != "ConceptGraphs"
        ]
        grouped_pct = grouped_pct.loc[strategy_order]

        colors = [
            "#ea9999",
            "#ffe599",
            "#b6d7a8",
        ]

        ax = grouped_pct[["False Count", "Partial Count", "True Count"]].plot(
            kind="bar", stacked=True, color=colors, figsize=(10, 7), width=0.7
        )
        plt.title("Response Quality Overall", fontweight="bold", fontsize=20)
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.set_xticklabels(
            [wrap_label(lbl.get_text(), width=14) for lbl in ax.get_xticklabels()],
            fontsize=18,
            rotation=0,
        )
        plt.legend(
            ["False", "Partial", "True"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=True,
            ncol=3,
            fontsize=16,
        )

        for c in ax.containers:
            ax.bar_label(
                c,
                fmt="%.1f%%",
                label_type="center",
                color="black",
                fontweight="bold",
                fontsize=18,
            )

        save_plot(plt.gcf(), "03_response_quality_stacked", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot response quality stacked bar: {e}")


def plot_04_overall_success_boxplot(
    df: pd.DataFrame, attrs: dict, out_dir: str
) -> None:
    """
    Plots a boxplot showing the distribution of overall success rates across homes for each strategy,
    with x-axis label ticks removed and a legend displayed in a box below the chart, centered.
    Ensures the legend is always present and not empty. Legend text is wrapped for readability.

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(6, 4))

        data = df[(df["Question Type"] == "Overall") & (df["Metric"] == "Success Rate")]

        ax = sns.boxplot(
            data=data,
            x="Strategy",
            y="Value",
            hue="Strategy",
            palette=get_palette(attrs),
            showfliers=False,
        )
        np.random.seed(100)
        sns.stripplot(
            data=data,
            x="Strategy",
            y="Value",
            color="black",
            alpha=0.5,
            ax=ax,
            jitter=True,
        )

        ax.set_ylim(0.3, 1)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_title("Distribution of Success Rates", fontweight="bold", pad=10, fontsize=14)
        ax.set_ylabel("Success Rate")

        handles, labels = ax.get_legend_handles_labels()
        if not handles or not labels:
            unique_strategies = data["Strategy"].unique()
            handles = [
                plt.Line2D([0], [0], color=color, lw=4) for color in get_palette(attrs)
            ]
            labels = [str(s) for s in unique_strategies]

        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.47, -0.2),
            frameon=True,
            ncol=len(labels),
            fontsize=10,
        )

        save_plot(plt.gcf(), "04_overall_success_boxplot", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot overall success boxplot: {e}")


def plot_05_radar_success_by_type(df: pd.DataFrame, attrs: dict, out_dir: str) -> None:
    """
    Generates a radar chart comparing strategies across question types, ensuring that axis labels are wrapped
    and the 'Graceful Failure' label receives extra spacing to prevent occlusion by the plot. The 'Original'
    strategy is always plotted first (leftmost in the legend and plot order).

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)

        data = df[(df["Question Type"] != "Overall") & (df["Metric"] == "Success Rate")]
        pivot = data.groupby(["Strategy", "Question Type"])["Value"].mean().unstack()

        strategy_order = ["ConceptGraphs"] + [s for s in pivot.index if s != "ConceptGraphs"]
        pivot = pivot.loc[strategy_order]

        categories = list(pivot.columns)
        N = len(categories)

        wrapped_categories = []
        for cat in categories:
            if cat == "Graceful Failure":
                wrapped = wrap_label(cat, 14) + "\n\n"
            else:
                wrapped = wrap_label(cat, 14)
            wrapped_categories.append(wrapped)

        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        palette = get_palette(attrs)
        if isinstance(palette, dict):
            colors = list(palette.values())
        else:
            colors = palette

        for i, (idx, row) in enumerate(pivot.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(
                angles,
                values,
                linewidth=2,
                linestyle="solid",
                label=idx,
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

        xtick_ys = [1.13, 1.08, 1.18, 1.08]
        for angle, label, y in zip(angles[:-1], wrapped_categories, xtick_ys):
            angle_shift = 0
            if "\n" in label:
                angle_shift = 0.05
            plt.text(
                angle + angle_shift,
                y,
                label,
                ha="center",
                va="center",
                size=18,
                fontweight="bold",
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(wrapped_categories, size=0)
        ax.set_title(
            "Distribution of Success Rates", fontweight="bold", pad=50, fontsize=24
        )

        ax.set_rlabel_position(0)
        plt.yticks(
            [0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=15
        )
        plt.ylim(0, 1)
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=True,
            ncol=len(pivot.index),
            fontsize=18,
        )

        save_plot(fig, "05_radar_success_by_type", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot radar chart for success by type: {e}")


def plot_06_heatmap_home_vs_strategy(
    df: pd.DataFrame, attrs: dict, out_dir: str
) -> None:
    """
    Plots a heatmap of overall success rates for each home and strategy, ensuring the 'Original'
    strategy is displayed as the leftmost column.

    :param df: Long-format DataFrame containing evaluation results.
    :type df: pd.DataFrame
    :param attrs: Dictionary of plot styling attributes.
    :type attrs: dict
    :param out_dir: Directory to save the plot.
    :type out_dir: str
    :raises KeyError: If required columns are missing.
    :raises ValueError: If data processing fails.
    :return: None
    :rtype: None
    """
    try:
        apply_plot_style(attrs)
        plt.figure(figsize=(10, 12))

        data = df[(df["Question Type"] == "Overall") & (df["Metric"] == "Success Rate")]
        pivot = data.pivot(index="home_id", columns="Strategy", values="Value")

        strategy_order = ["ConceptGraphs"] + [s for s in pivot.columns if s != "ConceptGraphs"]
        pivot = pivot[strategy_order]

        def wrap_text(text: str, width: int = 14) -> str:
            """
            Wraps the input text to the specified width.

            :param text: The text string to wrap.
            :type text: str
            :param width: Maximum line width before wrapping.
            :type width: int
            :return: Wrapped text string.
            :rtype: str
            """
            return "\n".join(textwrap.wrap(str(text), width=width))

        wrapped_columns = [wrap_text(col, 14) for col in pivot.columns]
        wrapped_index = [wrap_text(idx, 14) for idx in pivot.index]

        ax = sns.heatmap(
            pivot,
            annot=True,
            cmap="Purples",
            fmt=".2f",
            cbar_kws={"label": "Success Rate", "ticks": [0.4, 0.6, 0.8, 1.0]},
            annot_kws={"size": 20},
            vmin=0.4,
            vmax=1.0,
        )

        if ax.figure.axes and len(ax.figure.axes) > 1:
            colorbar = ax.figure.axes[-1]
            colorbar.set_yticks([0.4, 0.6, 0.8, 1.0])
            colorbar.set_ylim(0.4, 1.0)
            colorbar.tick_params(labelsize=20)

        ax.set_xticklabels(wrapped_columns, rotation=0, fontsize=16)
        ax.set_yticklabels(wrapped_index, rotation=0, fontsize=14)

        ax.set_title("Success Rate per Home", fontweight="bold", fontsize=32, pad=20)
        ax.set_ylabel("Home ID", fontsize=28, fontweight="bold")
        ax.set_xlabel("", fontsize=20)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=20)
        ax.figure.axes[-1].yaxis.label.set_size(28)
        ax.figure.axes[-1].yaxis.label.set_weight("bold")

        save_plot(plt.gcf(), "06_heatmap_home_vs_strategy", out_dir)
    except (KeyError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to plot heatmap for home vs strategy: {e}")


if __name__ == "__main__":

    DATABASE_PATH = os.path.join(
        "D:", "Documentos", "Datasets", "Robot@VirtualHomeLarge"
    )

    PATH_AK = os.path.join(
        DATABASE_PATH,
        "interaction_eval_results_with_ak",
        "summary_interaction_ak_eval.csv",
    )
    PATH_IMP = os.path.join(
        DATABASE_PATH, "interaction_eval_results", "summary_interaction_eval.csv"
    )
    PATH_ORIG = os.path.join(
        DATABASE_PATH,
        "original_interaction_eval_results",
        "summary_original_interaction_eval.csv",
    )

    OUTPUT_DIR = os.path.join(DATABASE_PATH, "evaluation_results", "plots_comparison")

    plot_attributes = {
        "style": "whitegrid",
        "font_family": "sans-serif",
        "font_scale": 1.1,
        "figsize": (12, 7),
        "title_size": 16,
        "label_size": 13,
        "dpi": 300,
        "palette": [
            "#07ca98",
            "#ef563c",
            "#6671fa",
        ],
    }

    PLOT_FUNCTIONS = {
        # "overall_success_bar": plot_01_overall_success_bar,
        "type_success_grouped": plot_02_type_success_grouped_bar,
        "response_quality_stacked": plot_03_response_quality_stacked,
        "overall_success_box": plot_04_overall_success_boxplot,
        "radar_success": plot_05_radar_success_by_type,
        "heatmap_home_strat": plot_06_heatmap_home_vs_strategy,
    }

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading and preprocessing data...")
        df_wide = load_and_preprocess_data(PATH_AK, PATH_IMP, PATH_ORIG)
        df_long = prepare_long_format(df_wide)
        print(f"Data loaded: {len(df_long)} records.")

        print(f"Generating {len(PLOT_FUNCTIONS)} plots...")
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            try:
                print(f"  -> {plot_name}...")
                plot_func(df_long, plot_attributes, OUTPUT_DIR)
            except (KeyError, ValueError, TypeError) as e:
                traceback.print_exc()
                print(f"  [ERROR] Failed to generate '{plot_name}': {e}")

        print("All plots generated successfully!")

    except (FileNotFoundError, ValueError) as e:
        traceback.print_exc()
        print(f"Critical error during execution: {e}")
