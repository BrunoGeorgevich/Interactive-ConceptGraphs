import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib as mpl
import pandas as pd
import numpy as np
import os


def apply_plot_style() -> None:
    """
    Applies the requested visual style to matplotlib plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 11
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300


COLORS: list[str] = [
    "#3498db",
    "#07ca98",
    "#ef563c",
    "#6671fa",
]


def process_single_experiment(args: tuple) -> dict | None:
    """
    Loads a log and calculates the simple mean for CPU, RAM, and GPU usage.
    Ignores network metrics and outliers.

    :param args: Tuple containing (outputs_dir, prefix, home_id, manager_dir, resource_log_name)
    :type args: tuple
    :raises FileNotFoundError: If the resource log file does not exist.
    :raises ValueError: If the CSV file is empty or missing required columns.
    :return: Dictionary with calculated statistics or None if data is invalid.
    :rtype: dict | None
    """
    outputs_dir, prefix, home_id, manager_dir, resource_log_name = args

    exp_folder = f"{prefix}_house_{home_id}_det"
    exp_path = os.path.join(
        outputs_dir, f"Home{home_id:02d}", "Wandering", "exps", exp_folder
    )
    csv_path = os.path.join(exp_path, manager_dir, resource_log_name)

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, sep=";")
        if df.empty or "timestamp" not in df.columns:
            return None

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="mixed", errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return None

        stats = {"prefix": prefix, "home_id": home_id}
        pct_cols = ["cpu_percent", "ram_percent", "gpu0_percent", "gpu0_mem_percent"]

        for col in pct_cols:
            if col in df.columns:
                stats[col] = df[col].mean()

        return stats

    except (pd.errors.ParserError, OSError, ValueError) as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(
            f"Failed to process experiment log for {exp_folder}: {e}"
        ) from e


def generate_bar_chart(df_stats: pd.DataFrame, output_path: str) -> None:
    """
    Generates a comparative bar chart (Mean of Houses ± Standard Deviation) for CPU, RAM, and GPU.

    :param df_stats: DataFrame containing statistics for each experiment.
    :type df_stats: pd.DataFrame
    :param output_path: Path to save the generated plot.
    :type output_path: str
    :raises OSError: If the plot cannot be saved to the specified path.
    """
    print(f"Generating chart: {output_path}...")
    apply_plot_style()

    prefixes = sorted(df_stats["prefix"].unique())
    preferred = ["original", "improved", "offline", "online"]
    prefixes = [p for p in preferred if p in prefixes]

    grouped_mean = df_stats.groupby("prefix").mean(numeric_only=True)
    grouped_std = df_stats.groupby("prefix").std(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    metrics = ["cpu_percent", "ram_percent", "gpu0_percent", "gpu0_mem_percent"]
    labels = ["CPU", "RAM", "GPU", "GPU Mem"]

    x = np.arange(len(metrics))
    width = 0.8 / len(prefixes)

    for i, prefix in enumerate(prefixes):
        if prefix not in grouped_mean.index:
            continue

        y_vals = [grouped_mean.loc[prefix, m] for m in metrics]
        y_errs = [grouped_std.loc[prefix, m] for m in metrics]

        pos = x + (i - len(prefixes) / 2 + 0.5) * width

        ax.bar(
            pos,
            y_vals,
            width,
            yerr=y_errs,
            label=prefix,
            capsize=4,
            color=COLORS[i % len(COLORS)],
            alpha=0.7,
        )

    ax.set_ylabel("Average Usage (%)", fontweight="bold")
    # ax.set_title("System Resources (Mean of Houses ± Std)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 80)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", alpha=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(prefixes),
        frameon=True,
    )

    for text in ax.get_legend().get_texts():
        text.set_text(text.get_text().capitalize())

    plt.tight_layout()
    try:
        plt.savefig(output_path, bbox_inches="tight")
    except (OSError, ValueError) as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Failed to save plot to {output_path}: {e}") from e
    plt.close(fig)
    print(f"Successfully saved: {output_path}")


def main() -> None:
    """
    Main function to process system resource logs, aggregate statistics, and generate plots.
    """
    DATABASE_PATH = os.path.join(
        "D:\\", "Documentos", "Datasets", "Robot@VirtualHomeLarge"
    )
    OUTPUTS_DIR = os.path.join(DATABASE_PATH, "outputs")
    PLOTS_DIR = os.path.join(DATABASE_PATH, "evaluation_results", "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PREFIXES = ["improved", "original", "offline", "online"]
    HOME_IDS = list(range(1, 31))
    MANAGER_DIR = "manager"
    LOG_NAME = "resource_log.csv"

    tasks = []
    for prefix in PREFIXES:
        for home_id in HOME_IDS:
            tasks.append((OUTPUTS_DIR, prefix, home_id, MANAGER_DIR, LOG_NAME))

    print(f"Starting parallel processing of {len(tasks)} logs...")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_single_experiment, tasks):
            if res is not None:
                results.append(res)

    print(f"Processing completed. {len(results)} logs analyzed.")

    if not results:
        print("No data found.")
        return

    df_stats = pd.DataFrame(results)

    csv_path = os.path.join(PLOTS_DIR, "system_resources_only.csv")
    df_stats.to_csv(csv_path, index=False, sep=";")
    print(f"Statistics CSV saved at: {csv_path}")

    # plot_path = os.path.join(PLOTS_DIR, "system_resources_comparison.png")
    plot_path = os.path.join(PLOTS_DIR, "system_resources_comparison.pdf")
    generate_bar_chart(df_stats, plot_path)


if __name__ == "__main__":
    main()
