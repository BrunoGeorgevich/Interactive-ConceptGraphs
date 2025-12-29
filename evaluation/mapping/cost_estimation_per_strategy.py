from dotenv import load_dotenv
from glob import glob
import traceback
import random
import json
import csv
import cv2
import os

from conceptgraph.inference.manager import AdaptiveInferenceManager
from conceptgraph.inference.cost_estimator import CostEstimator


def estimate_costs_per_house(
    house_stats: dict[int, int],
    estimated_costs: dict[str, dict[str, float]],
    strategies: list[str],
    stride: int,
) -> dict[str, dict[int, float]]:
    """
    Estimates the processing cost per house for each strategy.

    :param house_stats: Dictionary mapping house IDs to number of frames.
    :type house_stats: dict[int, int]
    :param estimator: The CostEstimator instance.
    :type estimator: CostEstimator
    :param strategies: List of strategy names.
    :type strategies: list[str]
    :param stride: The stride size for frame selection.
    :type stride: int
    :raises RuntimeError: If cost estimation fails for a strategy.
    :return: Nested dictionary with strategy as key and house cost mapping as value.
    :rtype: dict[str, dict[int, float]]
    """
    costs_per_house: dict[str, dict[int, float]] = {}
    for strategy in strategies:
        costs_per_house[strategy] = {}
        for home_id, num_frames in house_stats.items():
            try:
                num_rounds = (num_frames + stride - 1) // stride
                total_cost = num_rounds * estimated_costs[strategy]["cost_per_round"]
                costs_per_house[strategy][home_id] = float(f"{total_cost:.8f}")
            except (ValueError, ArithmeticError) as exc:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to estimate cost for house {home_id} with strategy {strategy}: {exc}"
                )
    return costs_per_house


def export_house_costs_to_csv(
    costs_per_house: dict[str, dict[int, float]],
    house_stats: dict[int, int],
    output_path: str,
) -> dict[int, dict[str, float]]:
    """
    Exports the estimated costs per house and strategy to a CSV file.

    :param costs_per_house: Nested dictionary with strategy as key and house cost mapping as value.
    :type costs_per_house: dict[str, dict[int, float]]
    :param house_stats: Dictionary mapping house IDs to number of frames.
    :type house_stats: dict[int, int]
    :param output_path: Path to the output CSV file.
    :type output_path: str
    :raises OSError: If the file cannot be written.
    :return: Dictionary mapping house IDs to their costs per strategy.
    :rtype: dict[int, dict[str, float]]
    """
    house_costs: dict[int, dict[str, float]] = {}

    try:
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            header = ["HouseID", "NumFrames"] + list(costs_per_house.keys())
            writer.writerow(header)
            for home_id in sorted(house_stats.keys()):
                num_frames = house_stats[home_id]
                row = [home_id, num_frames]
                house_costs[home_id] = {}
                for strategy in costs_per_house:
                    cost_per_house = costs_per_house[strategy].get(home_id, -1)
                    house_costs[home_id][strategy] = cost_per_house
                    row.append(f"{cost_per_house:.8f}")
                writer.writerow(row)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(f"Failed to write CSV file at {output_path}: {exc}")

    return house_costs


def compute_average_costs(
    house_costs: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Computes the average cost per house for each strategy.

    :param house_costs: Dictionary mapping house IDs to their costs per strategy.
    :type house_costs: dict[int, dict[str, float]]
    :raises ZeroDivisionError: If there are no houses to average.
    :return: Dictionary mapping strategy to average cost.
    :rtype: dict[str, float]
    """
    avg_costs: dict[str, float] = {}
    strategy_costs: dict[str, list[float]] = {}

    for home_id in house_costs:
        for strategy, cost in house_costs[home_id].items():
            if strategy not in strategy_costs:
                strategy_costs[strategy] = []
            strategy_costs[strategy].append(cost)

    for strategy, costs in strategy_costs.items():
        try:
            avg = sum(costs) / len(costs)
            avg_costs[strategy] = float(f"{avg:.8f}")
        except ZeroDivisionError as exc:
            traceback.print_exc()
            raise ZeroDivisionError(
                f"No houses to average for strategy {strategy}: {exc}"
            )
    return avg_costs


def process_image(img_path: str) -> None:
    """
    Processes a single image using the online_manager and estimates its cost.

    :param img_path: The path to the image file.
    :type img_path: str
    :raises FileNotFoundError: If the image file cannot be read.
    :raises ValueError: If the image path is invalid or processing fails.
    :return: None
    :rtype: None
    """
    try:
        image_np = cv2.imread(img_path)
        if image_np is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        manager.process_frame(img_path, image_np)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Skipping the frame: {img_path} due to error: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    load_dotenv()

    DATASET_BASE_PATH = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    STRIDE_SIZE = 20
    NUM_OF_SAMPLES = 30

    estimations_dir = "estimations"
    try:
        os.makedirs(estimations_dir, exist_ok=True)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(f"Failed to create estimations directory: {exc}")

    all_images: list[str] = []

    estimator = CostEstimator()
    estimator.register_model(
        "openai/gpt-4o", {"input_cost": 2.5 / 1_000_000, "output_cost": 10 / 1_000_000}
    )
    estimator.register_model(
        "gemini-2.5-flash-lite",
        {"input_cost": 0.3 / 1_000_000, "output_cost": 0.4 / 1_000_000},
    )
    estimator.register_model(
        "google/gemini-2.5-flash-lite",
        {"input_cost": 0.3 / 1_000_000, "output_cost": 0.4 / 1_000_000},
    )

    house_stats: dict[int, int] = {}

    for home_id in range(1, 31):
        home_path = os.path.join(DATASET_BASE_PATH, f"Home{home_id:02d}", "Wandering")
        rgb_images = glob(os.path.join(home_path, "*_rgb.jpg"))
        all_images.extend(rgb_images)
        house_stats[home_id] = len(rgb_images)

    random_chosen_images = random.choices(all_images, k=NUM_OF_SAMPLES)
    print(random_chosen_images)

    estimated_costs_json_path = os.path.join(estimations_dir, "estimated_costs.json")
    estimated_costs: dict[str, dict[str, float]] = {}

    if os.path.exists(estimated_costs_json_path):
        try:
            with open(estimated_costs_json_path, "r") as f:
                estimated_costs = json.load(f)
            print(
                f"Loaded estimated costs from {estimated_costs_json_path}: {estimated_costs}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to read estimated costs from {estimated_costs_json_path}: {exc}"
            )
    else:
        for strategy in ["original", "online", "improved"]:
            estimator.clear_executions()
            manager = AdaptiveInferenceManager(configuration=strategy)

            try:
                for img_path in random_chosen_images:
                    try:
                        process_image(img_path)
                    except (FileNotFoundError, ValueError, RuntimeError) as exc:
                        print(f"Error processing image: {exc}")
                        traceback.print_exc()
            except (OSError, RuntimeError) as exc:
                print(f"Image processing loop failed: {exc}")
                traceback.print_exc()

            total_cost = estimator.estimate_cost()
            cost_per_round = total_cost / NUM_OF_SAMPLES
            estimated_costs[strategy] = {
                "total_cost": float(f"{total_cost:.8f}"),
                "cost_per_round": float(f"{cost_per_round:.8f}"),
            }
            executions_csv_path = os.path.join(
                estimations_dir, f"executions_{strategy}.csv"
            )
            estimator.export_executions_to_csv(executions_csv_path)

        print(f"Estimated Costs per Strategy: {estimated_costs}")
        try:
            with open(estimated_costs_json_path, "w") as f:
                json.dump(estimated_costs, f, indent=4)
        except (OSError, PermissionError) as exc:
            traceback.print_exc()
            raise OSError(
                f"Failed to write estimated costs to {estimated_costs_json_path}: {exc}"
            )

    strategies = ["original", "online", "improved"]
    costs_per_house = estimate_costs_per_house(
        house_stats, estimated_costs, strategies, STRIDE_SIZE
    )
    house_costs_csv_path = os.path.join(estimations_dir, "house_costs.csv")
    house_costs = export_house_costs_to_csv(
        costs_per_house, house_stats, house_costs_csv_path
    )
    average_costs = compute_average_costs(house_costs)
    average_costs_json_path = os.path.join(estimations_dir, "average_costs.json")
    print(f"Average cost per house for each strategy: {average_costs}")
    try:
        with open(average_costs_json_path, "w") as f:
            json.dump(average_costs, f, indent=4)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(
            f"Failed to write average costs to {average_costs_json_path}: {exc}"
        )
