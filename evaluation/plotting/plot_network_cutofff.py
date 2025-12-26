import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os


def apply_plot_style() -> None:
    """
    Applies a consistent style to matplotlib plots for improved readability.

    :return: None
    :rtype: None
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300


COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]


def load_process_and_detect_cutoff(
    csv_path: str, max_seconds: int = 300, smoothing_window: int = 15
) -> tuple[pd.DataFrame | None, float, float | None]:
    """
    Loads a CSV file, processes resource usage metrics, and detects network cutoff events.

    :param csv_path: Path to the CSV file containing resource logs.
    :type csv_path: str
    :param max_seconds: Maximum time in seconds to include in the output DataFrame.
    :type max_seconds: int
    :param smoothing_window: Window size for rolling mean smoothing.
    :type smoothing_window: int
    :raises ValueError: If the CSV file is malformed or missing required columns.
    :return: Tuple containing processed DataFrame, last real elapsed time, and cutoff time.
    :rtype: tuple[pd.DataFrame | None, float, float | None]
    """
    if not os.path.exists(csv_path):
        return None, 0.0, None

    try:
        df = pd.read_csv(csv_path, sep=";")
        if df.empty or "timestamp" not in df.columns:
            return None, 0.0, None

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="mixed", errors="coerce"
        )
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if df.empty:
            return None, 0.0, None

        start_time = df["timestamp"].iloc[0]
        df["elapsed_seconds"] = (df["timestamp"] - start_time).dt.total_seconds()

        metric_cols = [
            "ram_percent",
            "gpu0_percent",
            "gpu0_mem_percent",
            "net_sent_mbps",
            "net_recv_mbps",
        ]
        cols_present = [c for c in metric_cols if c in df.columns]
        df = df[["elapsed_seconds"] + cols_present].copy()

        cutoff_time = None
        net_cols = ["net_sent_mbps", "net_recv_mbps"]
        present_net_cols = [c for c in net_cols if c in df.columns]

        if present_net_cols:
            mask_negative = (df[present_net_cols] < 0).any(axis=1)
            if mask_negative.any():
                cutoff_time = df.loc[mask_negative.idxmax(), "elapsed_seconds"]

        for col in present_net_cols:
            df[col] = df[col].apply(lambda x: max(0.0, x))

        if smoothing_window > 1:
            for col in cols_present:
                df[col] = df[col].rolling(window=smoothing_window, min_periods=1).mean()

        df = df[df["elapsed_seconds"] <= max_seconds]
        if df.empty:
            return None, 0.0, None

        last_real_time = df["elapsed_seconds"].iloc[-1]

        if round(last_real_time) < max_seconds:
            epsilon = 0.1
            pad_rows = [
                {"elapsed_seconds": last_real_time + epsilon},
                {"elapsed_seconds": max_seconds},
            ]
            for row in pad_rows:
                for col in cols_present:
                    row[col] = 0.0
            df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)

        return df, last_real_time, cutoff_time

    except (pd.errors.ParserError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
        import traceback

        traceback.print_exc()
        raise ValueError(f"Failed to process CSV file '{csv_path}': {e}")


def plot_strategies_comparison(
    outputs_dir: str,
    prefixes: list,
    home_id: int,
    output_dir: str,
    base_filename: str,
    max_seconds: int = 300,
    smoothing_window: int = 20,
) -> None:
    """
    Plots a 2x2 grid comparing resource usage metrics for different strategies.
    Generates one figure file per metric type.

    :param outputs_dir: Directory containing experiment outputs.
    :type outputs_dir: str
    :param prefixes: List of experiment prefixes to compare (Must be exactly 4 for 2x2 grid).
    :type prefixes: list
    :param home_id: Identifier for the home being analyzed.
    :type home_id: int
    :param output_dir: Directory to save the generated plots.
    :type output_dir: str
    :param base_filename: Base name for the output files (without extension).
    :type base_filename: str
    :param max_seconds: Maximum time in seconds for the x-axis.
    :type max_seconds: int
    :param smoothing_window: Window size for rolling mean smoothing.
    :type smoothing_window: int
    :return: None
    :rtype: None
    """
    print(f"Gerando grids 2x2 para Casa {home_id}...")
    apply_plot_style()

    if len(prefixes) != 4:
        print("Warning: This layout is optimized for exactly 4 prefixes (2x2 grid).")

    # 1. Load all data first to avoid reloading for every metric plot
    data_map = {}
    manager_dir = "manager"
    resource_log_name = "resource_log.csv"

    for prefix in prefixes:
        exp_folder = f"{prefix}_house_{home_id}_det"
        exp_path = os.path.join(
            outputs_dir, f"Home{home_id:02d}", "Wandering", "exps", exp_folder
        )
        csv_path = os.path.join(exp_path, manager_dir, resource_log_name)

        try:
            df, real_duration, cutoff_time = load_process_and_detect_cutoff(
                csv_path, max_seconds, smoothing_window
            )
            data_map[prefix] = {
                "df": df,
                "real_duration": real_duration,
                "cutoff_time": cutoff_time,
            }
        except ValueError:
            import traceback

            traceback.print_exc()
            data_map[prefix] = None

    metrics_config = [
        {
            "col": "ram_percent",
            "title": "RAM (%)",
            "color": COLORS[4],
            "ylim": (-2, 102),
        },
        {
            "col": "gpu0_percent",
            "title": "GPU (%)",
            "color": COLORS[2],
            "ylim": (-2, 102),
        },
        {
            "col": "gpu0_mem_percent",
            "title": "GPU Mem (%)",
            "color": COLORS[1],
            "ylim": (-2, 102),
        },
        {
            "col": "net_sent_mbps",
            "title": "Net Sent (Mbps)",
            "color": COLORS[3],
            "ylim": (-0.01, 3),
        },
        {
            "col": "net_recv_mbps",
            "title": "Net Recv (Mbps)",
            "color": COLORS[5],
            "ylim": (-0.001, 0.2),
        },
    ]

    # 2. Create one figure per metric (Strategy 2x2 grid)
    for config in metrics_config:
        metric_col = config["col"]
        metric_title = config["title"]
        print(f"  - Plotando métrica: {metric_title}...")

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        for idx, prefix in enumerate(prefixes):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            data = data_map.get(prefix)

            # Strategy Title
            clean_title = prefix.replace("lost_connection_", "").capitalize()
            ax.set_title(clean_title, fontweight="bold", fontsize=14)

            if data and data["df"] is not None and metric_col in data["df"].columns:
                df = data["df"]
                real_duration = data["real_duration"]
                cutoff_time = data["cutoff_time"]

                ax.plot(
                    df["elapsed_seconds"],
                    df[metric_col],
                    color=config["color"],
                    linewidth=2,
                    label=metric_title,
                )

                # Error Zone
                if round(real_duration) < max_seconds:
                    ax.axvspan(real_duration, max_seconds, facecolor="red", alpha=0.08)
                    ax.axvspan(
                        real_duration,
                        max_seconds,
                        facecolor="none",
                        edgecolor="red",
                        hatch="///",
                        alpha=0.3,
                        linewidth=0,
                    )
                    ax.text(
                        (real_duration + max_seconds) / 2,
                        (
                            config["ylim"][1]
                            if config["ylim"][1]
                            else df[metric_col].max()
                        )
                        * 0.5,
                        "ERROR",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=16,
                        fontweight="bold",
                        alpha=0.5,
                    )

                # Cutoff Line
                if cutoff_time is not None and cutoff_time <= max_seconds:
                    ax.axvline(
                        x=cutoff_time,
                        color="black",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                    )
                    # Label just for the first plot to avoid clutter or if specifically needed
                    y_pos = (
                        config["ylim"][1] if config["ylim"][1] else df[metric_col].max()
                    )
                    ax.text(
                        cutoff_time,
                        y_pos * 0.95,
                        "Cutoff  ",
                        ha="right",
                        va="top",
                        color="black",
                        fontweight="bold",
                        fontsize=12,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )

            # Grid Setup
            ax.set_xlim(0, max_seconds)
            ax.grid(axis="y", alpha=0.3)
            ax.grid(axis="x", alpha=0.3)
            if config["ylim"][1] is not None:
                ax.set_ylim(config["ylim"][0], config["ylim"][1])
            else:
                ax.set_ylim(bottom=config["ylim"][0])

        # Global Labels
        fig.supxlabel("Time (s)", fontsize=16, fontweight="bold")
        fig.supylabel(metric_title, fontsize=16, fontweight="bold")

        # Save specific file for this metric
        output_filename = f"{base_filename}_{metric_col}.pdf"
        final_path = os.path.join(output_dir, output_filename)
        plt.savefig(final_path, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"    Salvo: {final_path}")


def main() -> None:
    """
    Main entry point for generating and saving the resource usage comparison plots.

    :return: None
    :rtype: None
    """
    # DATABASE_PATH = THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET
    DATABASE_PATH = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
    OUTPUTS_DIR = os.path.join(DATABASE_PATH, "outputs")
    PLOTS_DIR = os.path.join(DATABASE_PATH, "evaluation_results", "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PREFIXES = [
        "lost_connection_improved",
        "lost_connection_original",
        "lost_connection_offline",
        "lost_connection_online",
    ]
    HOME_ID = 1

    # Base filename without extension or metric suffix
    BASE_FILENAME = "lost_connection_experiment"

    plot_strategies_comparison(
        OUTPUTS_DIR,
        PREFIXES,
        HOME_ID,
        PLOTS_DIR,  # Pass directory
        BASE_FILENAME,  # Pass base filename
        max_seconds=280,
        smoothing_window=60,
    )


if __name__ == "__main__":
    main()
