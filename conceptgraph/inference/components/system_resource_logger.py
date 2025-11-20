from datetime import datetime
import matplotlib.pyplot as plt
import threading
import traceback
import psutil
import pynvml
import torch
import time
import math
import csv
import os

import pandas as pd


class SystemResourceLogger:
    """
    Monitors and logs system resource usage (CPU, RAM, GPU, Network, Disk) in a background thread.

    This class is intended for use as a context manager to ensure proper startup and cleanup of logging resources.
    """

    def __init__(
        self,
        sample_interval: float = 0.5,
        output_path: str | None = None,
    ):
        """
        Initialize the SystemResourceLogger.

        :param sample_interval: Interval in seconds between resource measurements.
        :type sample_interval: float
        :param output_path: Path to save the CSV file. If None, uses default naming.
        :type output_path: str | None
        :raises ValueError: If sample_interval is not positive.
        """
        if sample_interval <= 0:
            raise ValueError("sample_interval must be a positive number.")

        self.sample_interval: float = sample_interval
        self.baseline_collection_flag: bool = True
        self.output_path: str = (
            output_path
            or f"resource_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        self.thread: threading.Thread | None = None
        self.running: bool = False
        self.cpu_count: int = psutil.cpu_count()

        self.gpu_available: bool = False
        try:
            pynvml.nvmlInit()
            self.gpu_count: int = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)
            ]
            self.gpu_available = True
        except (pynvml.NVMLError, OSError) as e:
            traceback.print_exc()
            print(f"GPU monitoring unavailable: {e}")

        self.initial_net_io = None
        self.last_net_io = None
        self.last_net_time = 0.0

        self.csv_file = None
        self.csv_writer = None

    def _get_current_resources(self) -> dict[str, float]:
        """
        Collect current system resource usage metrics.

        :return: Dictionary with current resource usage metrics.
        :rtype: dict[str, float]
        """
        resources: dict[str, float] = {}

        try:
            resources["baseline_collection"] = (
                1.0 if self.baseline_collection_flag else 0.0
            )
            cpu_percent_total = float(psutil.cpu_percent(interval=None))
            resources["cpu_percent"] = (
                cpu_percent_total / self.cpu_count if self.cpu_count > 0 else 0.0
            )

            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    resources["cpu_freq_mhz"] = float(cpu_freq.current)
                else:
                    resources["cpu_freq_mhz"] = 0.0
            except (AttributeError, NotImplementedError):
                resources["cpu_freq_mhz"] = 0.0

            mem = psutil.virtual_memory()
            resources["ram_used_mb"] = float(mem.used) / (1024 * 1024)
            resources["ram_percent"] = float(mem.percent)

            disk_io = psutil.disk_io_counters()
            resources["disk_read_mb"] = float(disk_io.read_bytes) / (1024 * 1024)
            resources["disk_write_mb"] = float(disk_io.write_bytes) / (1024 * 1024)

            current_net_io = psutil.net_io_counters()
            current_time = time.time()
            time_delta = current_time - self.last_net_time

            if time_delta > 0 and self.last_net_io:
                bytes_sent_delta = (
                    current_net_io.bytes_sent - self.last_net_io.bytes_sent
                )
                bytes_recv_delta = (
                    current_net_io.bytes_recv - self.last_net_io.bytes_recv
                )
                resources["net_sent_mbps"] = (bytes_sent_delta * 8 / time_delta) / (
                    1024 * 1024
                )
                resources["net_recv_mbps"] = (bytes_recv_delta * 8 / time_delta) / (
                    1024 * 1024
                )
            else:
                resources["net_sent_mbps"] = 0.0
                resources["net_recv_mbps"] = 0.0

            self.last_net_io = current_net_io
            self.last_net_time = current_time

            if self.gpu_available:
                for i, handle in enumerate(self.gpu_handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        resources[f"gpu{i}_percent"] = float(util.gpu)
                        resources[f"gpu{i}_mem_percent"] = float(util.memory)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        resources[f"gpu{i}_mem_used_mb"] = float(mem_info.used) / (
                            1024 * 1024
                        )
                    except (pynvml.NVMLError, OSError):
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to read GPU {i} resource usage.")

        except (psutil.Error, OSError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to collect system resources: {e}")

        return resources

    def _setup_csv(self, resource_keys: list[str]) -> None:
        """
        Set up the CSV file with appropriate headers.

        :param resource_keys: List of resource metric names to use as CSV headers.
        :type resource_keys: list[str]
        :raises OSError: If the CSV file cannot be created or written.
        :return: None
        :rtype: None
        """
        try:
            self.csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
            headers = ["timestamp"] + resource_keys
            self.csv_writer = csv.writer(self.csv_file, delimiter=";")
            self.csv_writer.writerow(headers)
            self.csv_file.flush()
        except OSError as e:
            traceback.print_exc()
            raise OSError(f"Failed to set up CSV file: {e}")

    def _log_resources(self) -> None:
        """
        Main logging loop that runs in a separate thread.

        :raises RuntimeError: If resource logging fails due to system errors.
        :return: None
        :rtype: None
        """
        try:
            self._get_current_resources()
            time.sleep(self.sample_interval)
            first_sample = self._get_current_resources()
            resource_keys = sorted(first_sample.keys())

            self._setup_csv(resource_keys)
            print(f"Starting resource logging (interval: {self.sample_interval}s)...")

            while self.running:
                loop_start_time = time.time()
                timestamp = datetime.now().isoformat()

                resources = self._get_current_resources()
                row = [timestamp]
                for key in resource_keys:
                    row.append(f"{resources.get(key, 0.0):.4f}")

                if self.csv_writer and self.csv_file:
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()

                loop_time = time.time() - loop_start_time
                sleep_time = self.sample_interval - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except (RuntimeError, OSError, ValueError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Resource logging failed: {e}")
        finally:
            self.running = False

    def start(self) -> None:
        """
        Start the resource logging in a separate thread.

        :raises RuntimeError: If the logger is already running.
        :return: None
        :rtype: None
        """
        if self.running:
            raise RuntimeError("SystemResourceLogger is already running.")

        self.initial_net_io = psutil.net_io_counters()
        self.last_net_io = self.initial_net_io
        self.last_net_time = time.time()

        self.running = True
        self.thread = threading.Thread(target=self._log_resources, daemon=True)
        self.thread.start()
        print(f"SystemResourceLogger started. Output: {self.output_path}")

    def set_baseline_collection_flag(self, flag: bool) -> None:
        """
        Set a flag to indicate whether the current logging phase is a baseline collection.

        :param flag: True if collecting baseline data, False otherwise.
        :type flag: bool
        :return: None
        :rtype: None
        """
        self.baseline_collection_flag = flag

    def _cleanup(self) -> None:
        """
        Stop the resource logging and perform cleanup.

        :return: None
        :rtype: None
        """
        if not self.running:
            print("SystemResourceLogger is not running.")
            return

        print("Stopping SystemResourceLogger...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=max(self.sample_interval * 2, 2.0))

        if self.csv_file:
            try:
                if not self.csv_file.closed:
                    self.csv_file.close()
            except OSError as e:
                print(f"Error closing CSV file: {e}")

        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except (pynvml.NVMLError, OSError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to shutdown GPU monitoring: {e}")

        print(f"Resource log saved to: {self.output_path}")
        self.thread = None
        self.csv_file = None
        self.csv_writer = None

    def plot_resources(
        self,
        output_image_path: str | None = None,
        dpi: int = 150,
        figure_width: int = 16,
        row_height: int = 4,
        line_width: float = 2.0,
        font_size_title: int = 20,
        font_size_labels: int = 18,
        font_size_ticks: int = 16,
        grid_alpha: float = 0.4,
        grid_color: str = "#cccccc",
        background_color: str = "white",
        subplot_wspace: float = 0.2,
        subplot_hspace: float = 0.3,
        baseline_hatch_color: str = "#e0e0e0",
        baseline_hatch_pattern: str = "//",
        baseline_hatch_alpha: float = 0.5,
    ) -> None:
        """
        Plot all logged resources in a single grid figure with two columns, using a professional and visually appealing style.
        Vertical dashed lines (alpha=0.3, black) indicate the end of the first and start of the last baseline_collection phase.
        Baseline collection regions are highlighted with a hatched background.

        :param output_image_path: Path to save the plot image. If None, uses default naming.
        :type output_image_path: str | None
        :param dpi: Resolution of the output image in dots per inch.
        :type dpi: int
        :param figure_width: Width of the figure in inches.
        :type figure_width: int
        :param row_height: Height of each row in inches.
        :type row_height: int
        :param line_width: Width of the plot lines.
        :type line_width: float
        :param font_size_title: Font size for subplot titles.
        :type font_size_title: int
        :param font_size_labels: Font size for axis labels.
        :type font_size_labels: int
        :param font_size_ticks: Font size for axis tick labels.
        :type font_size_ticks: int
        :param grid_alpha: Transparency level for grid lines (0.0 to 1.0).
        :type grid_alpha: float
        :param grid_color: Color of the grid lines.
        :type grid_color: str
        :param background_color: Background color for the plot.
        :type background_color: str
        :param subplot_wspace: Width space between subplots.
        :type subplot_wspace: float
        :param subplot_hspace: Height space between subplots.
        :type subplot_hspace: float
        :param baseline_hatch_color: Color for the baseline region hatch.
        :type baseline_hatch_color: str
        :param baseline_hatch_pattern: Matplotlib hatch pattern for baseline regions.
        :type baseline_hatch_pattern: str
        :param baseline_hatch_alpha: Alpha for the baseline region hatch (0.0 to 1.0).
        :type baseline_hatch_alpha: float
        :raises FileNotFoundError: If the CSV log file does not exist.
        :raises ValueError: If the CSV file is empty or has invalid data, or if parameters are invalid.
        :return: None
        :rtype: None
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"CSV log file not found: {self.output_path}")

        if dpi <= 0:
            raise ValueError("dpi must be a positive integer.")
        if figure_width <= 0:
            raise ValueError("figure_width must be a positive integer.")
        if row_height <= 0:
            raise ValueError("row_height must be a positive integer.")
        if line_width <= 0:
            raise ValueError("line_width must be a positive number.")
        if font_size_title <= 0:
            raise ValueError("font_size_title must be a positive integer.")
        if font_size_labels <= 0:
            raise ValueError("font_size_labels must be a positive integer.")
        if font_size_ticks <= 0:
            raise ValueError("font_size_ticks must be a positive integer.")
        if not 0.0 <= grid_alpha <= 1.0:
            raise ValueError("grid_alpha must be between 0.0 and 1.0.")
        if subplot_wspace < 0.0:
            raise ValueError("subplot_wspace must be non-negative.")
        if subplot_hspace < 0.0:
            raise ValueError("subplot_hspace must be non-negative.")
        if not 0.0 <= baseline_hatch_alpha <= 1.0:
            raise ValueError("baseline_hatch_alpha must be between 0.0 and 1.0.")

        try:
            plt.style.use("seaborn-v0_8-darkgrid")
            df = pd.read_csv(self.output_path, sep=";")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            traceback.print_exc()
            raise ValueError(f"Failed to read CSV file: {e}")

        if df.empty:
            raise ValueError("CSV file is empty, cannot plot resources.")

        if "timestamp" not in df.columns:
            raise ValueError("CSV file missing 'timestamp' column.")

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            start_time = df["timestamp"].iloc[0]
            df["elapsed_seconds"] = (df["timestamp"] - start_time).dt.total_seconds()
        except (ValueError, TypeError, AttributeError) as e:
            traceback.print_exc()
            raise ValueError(f"Invalid timestamp format in CSV: {e}")

        plot_configs = [
            {
                "col_name": "cpu_percent",
                "title": "CPU Usage (%)",
                "ylabel": "CPU %",
                "y_min": -2,
                "y_max": 102,
                "color": "#1f77b4",
            },
            {
                "col_name": "ram_percent",
                "title": "RAM Usage (%)",
                "ylabel": "RAM %",
                "y_min": -2,
                "y_max": 102,
                "color": "#ff7f0e",
            },
            {
                "col_name": "gpu0_percent",
                "title": "GPU Usage (%)",
                "ylabel": "GPU %",
                "y_min": -2,
                "y_max": 102,
                "color": "#2ca02c",
            },
            {
                "col_name": "gpu0_mem_percent",
                "title": "GPU Memory Usage (%)",
                "ylabel": "GPU Mem %",
                "y_min": -2,
                "y_max": 102,
                "color": "#d62728",
            },
            {
                "col_name": "net_sent_mbps",
                "title": "Network Sent (Mbps)",
                "ylabel": "Mbps",
                "y_min": -0.05,
                "y_max": None,
                "color": "#9467bd",
            },
            {
                "col_name": "net_recv_mbps",
                "title": "Network Received (Mbps)",
                "ylabel": "Mbps",
                "y_min": -0.05,
                "y_max": None,
                "color": "#8c564b",
            },
        ]

        num_plots = len(plot_configs)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        baseline_col = "baseline_collection"
        baseline_regions = []
        baseline_indices = []
        if baseline_col in df.columns:
            baseline_flags = df[baseline_col].astype(float).values
            transitions = []
            for i in range(1, len(baseline_flags)):
                if baseline_flags[i - 1] == 1.0 and baseline_flags[i] == 0.0:
                    transitions.append(i)
                if baseline_flags[i - 1] == 0.0 and baseline_flags[i] == 1.0:
                    transitions.append(i)
            baseline_indices = transitions

            # Find baseline regions (start, end) in elapsed_seconds
            in_baseline = baseline_flags[0] == 1.0
            region_start = 0 if in_baseline else None
            for i in range(1, len(baseline_flags)):
                if not in_baseline and baseline_flags[i] == 1.0:
                    region_start = i
                    in_baseline = True
                elif in_baseline and baseline_flags[i] == 0.0:
                    region_end = i
                    baseline_regions.append((region_start, region_end))
                    in_baseline = False
            if in_baseline and region_start is not None:
                baseline_regions.append((region_start, len(baseline_flags)))

        try:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(figure_width, row_height * num_rows),
                constrained_layout=False,
            )
            fig.patch.set_facecolor(background_color)
            plt.subplots_adjust(wspace=subplot_wspace, hspace=subplot_hspace)
            if num_rows == 1:
                axes = [axes]
            axes = [
                ax
                for row in axes
                for ax in (row if hasattr(row, "__iter__") else [row])
            ]

            for idx, config in enumerate(plot_configs):
                ax = axes[idx]
                ax.set_facecolor(background_color)
                current_row = idx // num_cols
                is_last_row = current_row == num_rows - 1

                col_name = config["col_name"]
                title = config["title"]
                ylabel = config["ylabel"]
                y_min = config["y_min"]
                y_max = config["y_max"]
                color = config["color"]

                # Draw baseline regions as hatched rectangles
                for region in baseline_regions:
                    start_idx, end_idx = region
                    if 0 <= start_idx < len(df["elapsed_seconds"]):
                        x_start = df["elapsed_seconds"].iloc[start_idx]
                    else:
                        x_start = df["elapsed_seconds"].iloc[0]
                    if 0 < end_idx <= len(df["elapsed_seconds"]):
                        x_end = df["elapsed_seconds"].iloc[end_idx - 1]
                    else:
                        x_end = df["elapsed_seconds"].iloc[-1]
                    ax.axvspan(
                        x_start,
                        x_end,
                        facecolor=baseline_hatch_color,
                        alpha=baseline_hatch_alpha,
                        hatch=baseline_hatch_pattern,
                        edgecolor="none",
                        zorder=0,
                    )

                if col_name in df.columns:
                    ax.plot(
                        df["elapsed_seconds"],
                        df[col_name],
                        label=title,
                        color=color,
                        linewidth=line_width,
                        alpha=0.95,
                    )
                    ax.set_title(
                        title, fontsize=font_size_title, fontweight="bold", pad=14
                    )
                    ax.set_ylabel(
                        ylabel, fontsize=font_size_labels, fontweight="semibold"
                    )
                    ax.set_ylim(bottom=y_min, top=y_max)
                    ax.grid(True, alpha=grid_alpha, linestyle="--", color=grid_color)
                    ax.tick_params(axis="both", labelsize=font_size_ticks)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    for idx_transition in baseline_indices:
                        if 0 <= idx_transition < len(df["elapsed_seconds"]):
                            x = df["elapsed_seconds"].iloc[idx_transition]
                            ax.axvline(
                                x=x,
                                color="black",
                                linestyle="--",
                                alpha=0.3,
                                linewidth=2,
                                zorder=1,
                            )

                    if is_last_row:
                        ax.set_xlabel(
                            "Elapsed Time (seconds)",
                            fontsize=font_size_labels,
                            fontweight="semibold",
                        )
                        ax.tick_params(axis="x", rotation=0)
                    else:
                        ax.set_xticklabels([])
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {col_name}",
                        ha="center",
                        va="center",
                        fontsize=font_size_labels,
                        color="gray",
                        fontweight="semibold",
                        alpha=0.7,
                    )
                    ax.set_title(
                        title, fontsize=font_size_title, fontweight="bold", pad=14
                    )
                    if is_last_row:
                        ax.set_xlabel(
                            "Elapsed Time (seconds)",
                            fontsize=font_size_labels,
                            fontweight="semibold",
                        )

            for idx in range(len(plot_configs), len(axes)):
                axes[idx].axis("off")

            if output_image_path is None:
                output_image_path = self.output_path.replace(".csv", "_plot.png")

            plt.savefig(
                output_image_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=background_color,
            )
            print(f"Resource plot saved to: {output_image_path}")
            plt.close(fig)

        except (ValueError, OSError, TypeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate resource plot: {e}")

    def __enter__(self) -> "SystemResourceLogger":
        """
        Start the logger upon entering the context manager.

        :return: The SystemResourceLogger instance.
        :rtype: SystemResourceLogger
        """
        self.start()
        return self

    def __exit__(
        self, exc_type: type | None, exc_value: object | None, tb: object | None
    ) -> None:
        """
        Stop and clean up the logger upon exiting the context manager.

        :param exc_type: Exception type, if any.
        :type exc_type: type | None
        :param exc_value: Exception value, if any.
        :type exc_value: object | None
        :param tb: Traceback object, if any.
        :type tb: object | None
        :return: None
        :rtype: None
        """
        self._cleanup()


if __name__ == "__main__":

    print("Iniciando o programa...")

    try:
        with SystemResourceLogger(sample_interval=0.01) as logger:
            print("Rodando a 'fase 1' (ex: baseline) por 3 segundos...")
            time.sleep(3)
            logger.set_baseline_collection_flag(False)

            print("Rodando a 'fase 2' (ex: workload) por 5 segundos...")

            def cpu_intensive_task(duration: float) -> None:
                """
                Perform a highly CPU-intensive computation for a specified duration.

                :param duration: Duration in seconds to run the computation.
                :type duration: float
                :raises ValueError: If duration is not positive.
                :return: None
                :rtype: None
                """

                if duration <= 0:
                    raise ValueError("Duration must be positive.")

                start_time = time.time()
                try:
                    while time.time() - start_time < duration:
                        _ = sum(
                            math.sqrt(i) * math.log(i + 1) * math.sin(i)
                            for i in range(1, 100000)
                        )
                except (ArithmeticError, OverflowError) as e:
                    traceback.print_exc()
                    raise RuntimeError(f"CPU-intensive task failed: {e}")

            def gpu_intensive_task(duration: float) -> None:
                """
                Perform a GPU-intensive computation for a specified duration if GPU is available.

                :param duration: Duration in seconds to run the computation.
                :type duration: float
                :raises ValueError: If duration is not positive.
                :return: None
                :rtype: None
                """
                if not torch.cuda.is_available():
                    return

                if duration <= 0:
                    raise ValueError("Duration must be positive.")

                device = torch.device("cuda")
                start_time = time.time()
                try:
                    while time.time() - start_time < duration:
                        a = torch.randn(1024, 1024, device=device)
                        b = torch.randn(1024, 1024, device=device)
                        _ = torch.mm(a, b)
                        torch.cuda.synchronize()
                except (RuntimeError, ValueError) as e:
                    traceback.print_exc()
                    raise RuntimeError(f"GPU-intensive task failed: {e}")

            cpu_thread = threading.Thread(target=cpu_intensive_task, args=(5,))
            gpu_thread = threading.Thread(target=gpu_intensive_task, args=(5,))

            cpu_thread.start()
            gpu_thread.start()

            cpu_thread.join()
            gpu_thread.join()

            logger.set_baseline_collection_flag(True)
            print("Rodando a 'fase 3' (ex: baseline) por 3 segundos...")
            time.sleep(3)

        logger.plot_resources()

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")

    print("Programa finalizado.")
