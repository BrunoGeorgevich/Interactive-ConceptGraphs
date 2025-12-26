from multiprocessing import process
from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import traceback
import os


def load_summary_csv(summary_path: str) -> pd.DataFrame | None:
    """
    Loads the summary CSV file containing evaluation metrics.

    :param summary_path: Path to the summary CSV file.
    :type summary_path: str
    :raises FileNotFoundError: If the summary file does not exist.
    :raises ValueError: If the CSV file is malformed or empty.
    :return: DataFrame with evaluation metrics or None if loading fails.
    :rtype: pd.DataFrame | None
    """
    if not summary_path:
        raise ValueError("Summary path cannot be empty.")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    try:
        df = pd.read_csv(summary_path, delimiter=";")
        if df.empty:
            raise ValueError("Summary CSV file is empty.")
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        traceback.print_exc()
        raise ValueError(f"Error parsing summary CSV: {e}")


def load_detailed_csv(detailed_path: str) -> pd.DataFrame | None:
    """
    Loads a detailed comparisons CSV file.

    :param detailed_path: Path to the detailed CSV file.
    :type detailed_path: str
    :raises FileNotFoundError: If the detailed file does not exist.
    :raises ValueError: If the CSV file is malformed or empty.
    :return: DataFrame with detailed comparison data or None if loading fails.
    :rtype: pd.DataFrame | None
    """
    if not detailed_path:
        raise ValueError("Detailed path cannot be empty.")

    if not os.path.exists(detailed_path):
        raise FileNotFoundError(f"Detailed file not found: {detailed_path}")

    try:
        df = pd.read_csv(detailed_path, delimiter=";")
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        traceback.print_exc()
        raise ValueError(f"Error parsing detailed CSV: {e}")


def load_objects_csv(objects_path: str) -> pd.DataFrame | None:
    """
    Loads an objects CSV file containing voting results.

    :param objects_path: Path to the objects CSV file.
    :type objects_path: str
    :raises FileNotFoundError: If the objects file does not exist.
    :raises ValueError: If the CSV file is malformed or empty.
    :return: DataFrame with object voting data or None if loading fails.
    :rtype: pd.DataFrame | None
    """
    if not objects_path:
        raise ValueError("Objects path cannot be empty.")

    if not os.path.exists(objects_path):
        raise FileNotFoundError(f"Objects file not found: {objects_path}")

    try:
        df = pd.read_csv(objects_path, delimiter=";")
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        traceback.print_exc()
        raise ValueError(f"Error parsing objects CSV: {e}")


def apply_plot_style(plot_attributes: dict) -> None:
    """
    Applies global matplotlib styling based on plot attributes.

    :param plot_attributes: Dictionary containing plot styling attributes.
    :type plot_attributes: dict
    :return: None
    :rtype: None
    """
    plt.style.use(plot_attributes.get("style", "seaborn-v0_8-whitegrid"))
    mpl.rcParams["font.family"] = plot_attributes.get("font_family", "sans-serif")
    mpl.rcParams["font.size"] = plot_attributes.get("font_size", 12)
    mpl.rcParams["axes.titlesize"] = plot_attributes.get("title_size", 16)
    mpl.rcParams["axes.labelsize"] = plot_attributes.get("label_size", 14)
    mpl.rcParams["xtick.labelsize"] = plot_attributes.get("tick_size", 11)
    mpl.rcParams["ytick.labelsize"] = plot_attributes.get("tick_size", 11)
    mpl.rcParams["legend.fontsize"] = plot_attributes.get("legend_size", 11)
    mpl.rcParams["figure.dpi"] = plot_attributes.get("dpi", 150)
    mpl.rcParams["savefig.dpi"] = plot_attributes.get("save_dpi", 300)
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def get_color_palette(plot_attributes: dict) -> list:
    """
    Returns the color palette from plot attributes.

    :param plot_attributes: Dictionary containing plot styling attributes.
    :type plot_attributes: dict
    :return: List of colors for plotting.
    :rtype: list
    """
    return plot_attributes.get(
        "color_palette",
        ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"],
    )


def plot_metrics_boxplot(
    df: pd.DataFrame, plot_attributes: dict, output_dir: str
) -> None:
    """
    Creates boxplots showing distribution of metrics across homes for each processing type.
    Generates both a combined plot with all three metrics and individual plots for each metric.

    :param df: DataFrame with evaluation metrics.
    :type df: pd.DataFrame
    :param plot_attributes: Dictionary containing plot styling attributes.
    :type plot_attributes: dict
    :param output_dir: Directory to save the plot.
    :type output_dir: str
    :return: None
    :rtype: None
    :raises KeyError: If required columns are missing in the DataFrame.
    :raises ValueError: If DataFrame is empty or processing types are not found.
    """
    import matplotlib.patches
    
    apply_plot_style(plot_attributes)
    colors = get_color_palette(plot_attributes)

    try:
        metrics = ["precision", "recall", "f1_score"]
        titles = ["Precision", "Recall", "F1-Score"]

        processing_types = plot_attributes.get(
            "processing_type_order", df["processing_type"].unique()
        )
        processing_types = [
            pt for pt in processing_types if pt in df["processing_type"].values
        ]

        if df.empty or not processing_types:
            raise ValueError("DataFrame is empty or no valid processing types found.")

        legend_handles = [
            matplotlib.patches.Patch(
                facecolor=colors[i],
                edgecolor="black",
                label=processing_types[i].title(),
                alpha=0.7,
            )
            for i in range(len(processing_types))
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            data = [
                df[df["processing_type"] == pt][metric].values
                for pt in processing_types
            ]

            bp = ax.boxplot(
                data,
                patch_artist=True,
                labels=processing_types,
                showfliers=False,
                widths=0.8,
            )

            for patch, color in zip(bp["boxes"], colors[: len(processing_types)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.6)

            ax.set_ylim(0.2, 0.9)
            ax.set_xticklabels([])
            ax.set_yticklabels(
                [str(tick) for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
                fontsize=14,
            )
            ax.grid(axis="y", alpha=0.75)
            ax.grid(axis="x", alpha=0)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_title(title, fontweight="bold", pad=10)
            if idx == 0:
                ax.set_ylabel("Score", fontweight="bold", labelpad=10)

        plt.tight_layout(rect=[0, 0.15, 1, 1])
        plt.figlegend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
            fontsize=15,
            ncol=4,
        )
        fig.subplots_adjust(wspace=0.15, right=0.97)
        save_path = os.path.join(output_dir, "metrics_boxplot.pdf")
        plt.savefig(save_path, facecolor="white")
        plt.close()
        print(f"Saved: {save_path}")

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 4))

            data = [
                df[df["processing_type"] == pt][metric].values
                for pt in processing_types
            ]

            bp = ax_single.boxplot(
                data,
                patch_artist=True,
                labels=processing_types,
                showfliers=False,
                widths=0.8,
            )

            for patch, color in zip(bp["boxes"], colors[: len(processing_types)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.6)

            ax_single.set_ylim(0.2, 0.9)
            ax_single.set_xticklabels([])
            ax_single.set_yticklabels(
                [str(tick) for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
                fontsize=14,
            )
            ax_single.grid(axis="y", alpha=0.75)
            ax_single.grid(axis="x", alpha=0)
            ax_single.spines["top"].set_visible(False)
            ax_single.spines["left"].set_visible(False)
            ax_single.spines["bottom"].set_visible(False)
            ax_single.set_title(title, fontweight="bold", pad=10)
            ax_single.set_ylabel("Score", fontweight="bold", labelpad=10)

            if metric == "f1_score":
                plt.tight_layout(rect=[0, 0.15, 1, 1])
                plt.figlegend(
                    handles=legend_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    frameon=True,
                    fontsize=15,
                    ncol=2,
                )
            else:
                plt.tight_layout()

            save_path_single = os.path.join(output_dir, f"{metric}_boxplot.pdf")
            plt.savefig(save_path_single, facecolor="white")
            plt.close()
            print(f"Saved: {save_path_single}")

    except (KeyError, ValueError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Error generating metrics boxplot: {e}")


def plot_tp_fn_fp_grouped_bar(
    df: pd.DataFrame, plot_attributes: dict, output_dir: str
) -> None:
    """
    Creates a grouped horizontal bar chart showing TP, FN, FP side by side for each processing type.

    :param df: DataFrame with evaluation metrics.
    :type df: pd.DataFrame
    :param plot_attributes: Dictionary containing plot styling attributes.
    :type plot_attributes: dict
    :param output_dir: Directory to save the plot.
    :type output_dir: str
    :return: None
    :rtype: None
    :raises KeyError: If required columns are missing in the DataFrame.
    :raises ValueError: If DataFrame is empty or processing types are not found.
    """
    apply_plot_style(plot_attributes)

    try:
        processing_types = plot_attributes.get(
            "processing_type_order", df["processing_type"].unique()
        )
        processing_types = [
            pt for pt in processing_types if pt in df["processing_type"].values
        ][::-1]

        if df.empty or not processing_types:
            raise ValueError("DataFrame is empty or no valid processing types found.")

        tp_means = [
            df[df["processing_type"] == pt]["tp"].mean() for pt in processing_types
        ]
        fn_means = [
            df[df["processing_type"] == pt]["fn"].mean() for pt in processing_types
        ]
        fp_means = [
            df[df["processing_type"] == pt]["fp"].mean() for pt in processing_types
        ]

        _, ax = plt.subplots(figsize=(9, 5))

        y = range(len(processing_types))
        height = 0.3

        bars_fp = ax.barh(
            [i + height for i in y],
            fp_means,
            height,
            label="False Positives",
            color="#ea9999",
            edgecolor="black",
            linewidth=0.0,
        )

        bars_tp = ax.barh(
            y,
            tp_means,
            height,
            label="True Positives",
            color="#b6d7a8",
            edgecolor="black",
            linewidth=0.0,
        )
        bars_fn = ax.barh(
            [i - height for i in y],
            fn_means,
            height,
            label="False Negatives",
            color="#fade8b",
            edgecolor="black",
            linewidth=0.0,
        )

        for bars in [bars_tp, bars_fn, bars_fp]:
            for bar in bars:
                width_val = bar.get_width()
                ax.annotate(
                    f"{width_val:.1f}",
                    xy=(width_val + 1, bar.get_y() + bar.get_height() / 2),
                    va="center",
                    ha="left",
                    fontsize=17,
                    fontweight="bold",
                )

        ax.set_ylabel("", fontweight="bold", fontsize=18)
        ax.set_xlabel("Average Count", fontweight="bold", fontsize=22)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)
        ax.set_yticks(y)
        ax.set_yticklabels([p.title() for p in processing_types], fontsize=18)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            frameon=True,
            fontsize=18,
            ncol=3,
        )
        ax.grid(axis="x", alpha=0)
        ax.grid(axis="y", alpha=0)

        plt.tight_layout()
        # save_path = os.path.join(output_dir, "tp_fn_fp_grouped_bar.png")
        save_path = os.path.join(output_dir, "tp_fn_fp_grouped_bar.pdf")
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {save_path}")
    except (KeyError, ValueError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Error generating grouped horizontal bar plot: {e}")


def generate_all_plots(
    summary_df: pd.DataFrame,
    plot_attributes: dict,
    output_dir: str,
    plot_functions: dict,
) -> None:
    """
    Generates all enabled plots using the provided plotting functions dictionary.

    :param summary_df: DataFrame with evaluation metrics.
    :type summary_df: pd.DataFrame
    :param plot_attributes: Dictionary containing plot styling attributes.
    :type plot_attributes: dict
    :param output_dir: Directory to save the plots.
    :type output_dir: str
    :param plot_functions: Dictionary mapping plot names to their generator functions.
    :type plot_functions: dict
    :return: None
    :rtype: None
    """
    os.makedirs(output_dir, exist_ok=True)

    for plot_name, plot_func in plot_functions.items():
        try:
            print(f"Generating: {plot_name}...")
            plot_func(summary_df, plot_attributes, output_dir)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            traceback.print_exc()
            print(f"Error generating {plot_name}: {e}")


if __name__ == "__main__":
    """
    Main entry point for generating evaluation result plots.
    """
    # DATABASE_PATH: str = THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET
    DATABASE_PATH: str = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
    SUMMARY_CSV_PATH: str = os.path.join(
        DATABASE_PATH, "evaluation_results", "summary_all_homes.csv"
    )
    OUTPUT_PLOTS_DIR: str = os.path.join(DATABASE_PATH, "evaluation_results", "plots")

    plot_attributes: dict = {
        "style": "seaborn-v0_8-whitegrid",
        "font_family": "sans-serif",
        "font_size": 14,
        "title_size": 18,
        "label_size": 16,
        "tick_size": 12,
        "legend_size": 12,
        "dpi": 150,
        "save_dpi": 300,
        "figsize": (12, 7),
        "heatmap_figsize": (10, 12),
        "boxplot_figsize": (16, 6),
        "radar_figsize": (10, 10),
        "table_figsize": (14, 5),
        "color_palette": [
            "#3498db",
            "#07ca98",
            "#ef563c",
            "#6671fa",
            "#f39c12",
            "#1abc9c",
        ],
        "heatmap_cmap": "viridis",
        "processing_type_order": ["original", "improved", "offline", "online"],
        "baseline_type": "original",
    }

    PLOT_FUNCTIONS: dict = {
        "metrics_boxplot": plot_metrics_boxplot,
        "tp_fn_fp_grouped_bar": plot_tp_fn_fp_grouped_bar,
    }

    try:
        summary_df = load_summary_csv(SUMMARY_CSV_PATH)

        if summary_df is not None:
            print(f"Loaded summary with {len(summary_df)} rows.")
            print(
                f"Processing types found: {summary_df['processing_type'].unique().tolist()}"
            )
            print(f"Homes found: {sorted(summary_df['home_id'].unique().tolist())}")
            print("-" * 60)

            generate_all_plots(
                summary_df=summary_df,
                plot_attributes=plot_attributes,
                output_dir=OUTPUT_PLOTS_DIR,
                plot_functions=PLOT_FUNCTIONS,
            )

            print("-" * 60)
            print(f"All plots saved to: {OUTPUT_PLOTS_DIR}")
        else:
            print("Failed to load summary data.")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        traceback.print_exc()
        print(f"Error during plot generation: {e}")
