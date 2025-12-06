from google.genai.types import GenerateContentResponse
from agno.run.response import RunResponse
from typing import Union
import traceback


class CostEstimator:
    """
    Singleton class for estimating costs in concept graphs.

    This class ensures only one instance exists throughout the application,
    providing a centralized interface for cost estimation logic.

    :raises RuntimeError: If an attempt is made to instantiate more than one instance.
    """

    _instance: Union["CostEstimator", None] = None

    def __new__(cls) -> "CostEstimator":
        """
        Creates or returns the singleton instance of CostEstimator.

        :raises RuntimeError: If an attempt is made to instantiate more than one instance.
        :return: The singleton instance of CostEstimator.
        :rtype: CostEstimator
        """

        if cls._instance is None:
            try:
                cls._instance = super().__new__(cls)
            except (TypeError, ValueError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to create CostEstimator singleton: {e}")
        return cls._instance

    def __init__(self) -> None:
        """
        Initializes the CostEstimator singleton instance.

        :raises RuntimeError: If an attempt is made to reinitialize the singleton instance.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized: bool = True
        self._models = {}
        self._executions = []

    def register_model(self, model_name: str, model_info: dict) -> None:
        """
        Registers a model with its associated information.

        :param model_name: The name of the model to register.
        :type model_name: str
        :param model_info: A dictionary containing information about the model.
        :type model_info: dict
        """
        self._models[model_name] = model_info

    def register_execution(
        self,
        execution_type: str,
        execution_data: Union[dict, RunResponse, GenerateContentResponse],
    ) -> None:
        """
        Registers an execution with its associated information.

        :param execution_type: The type of execution (e.g., "detection", "classification").
        :type execution_type: str
        :param execution_data: A dictionary or RunResponse object containing execution details.
        :type execution_data: Union[dict, RunResponse, GenerateContentResponse]
        """

        execution_info = {}
        if isinstance(execution_data, RunResponse):
            execution_info = {
                "type": execution_type,
                "model": execution_data.model,
                "input_tokens": execution_data.metrics["input_tokens"][0],
                "output_tokens": execution_data.metrics["output_tokens"][0],
                "cached_tokens": execution_data.metrics.get("cached_tokens", [0])[0],
            }
        elif isinstance(execution_data, GenerateContentResponse):
            execution_info = {
                "type": execution_type,
                "model": execution_data.model_version,
                "input_tokens": execution_data.usage_metadata.prompt_token_count,
                "output_tokens": execution_data.usage_metadata.candidates_token_count,
                "cached_tokens": getattr(
                    execution_data.usage_metadata, "cached_content_token_count", 0
                ),
            }
        elif isinstance(execution_data, dict):
            execution_data["type"] = execution_type
            execution_info = execution_data
        else:
            raise TypeError(f"Unsupported execution_data type: {type(execution_data)}")

        self._executions.append(execution_info)

    def estimate_cost(self) -> float:
        """
        Estimates the total cost based on registered models and executions.

        :return: The estimated total cost.
        :rtype: float
        """
        total_cost = 0.0
        for ex in self._executions:
            model_name = ex.get("model")
            model_info = self._models.get(model_name, {})
            input_cost = model_info.get("input_cost", 0.0)
            output_cost = model_info.get("output_cost", 0.0)
            input_tokens = ex.get("input_tokens", 0)
            output_tokens = ex.get("output_tokens", 0)

            total_cost += (input_cost * input_tokens) + (output_cost * output_tokens)
        return total_cost

    def export_executions_to_csv(self, file_path: str = "executions.csv") -> None:
        """
        Exports the registered executions to a CSV file, including input_cost and output_cost columns.

        :param file_path: The path to the CSV file where executions will be saved.
        :type file_path: str
        :raises ValueError: If there are no executions to export.
        :raises RuntimeError: If writing to the CSV file fails.
        :return: None
        :rtype: None
        """
        import csv

        if not self._executions:
            raise ValueError("No executions to export.")

        fieldnames = list(self._executions[0].keys())
        if "input_cost" not in fieldnames:
            fieldnames.append("input_cost")
        if "output_cost" not in fieldnames:
            fieldnames.append("output_cost")

        try:
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                for execution in self._executions:
                    model_name = execution.get("model")
                    model_info = self._models.get(model_name, {})
                    row = dict(execution)
                    row["input_cost"] = model_info.get("input_cost", -1) * row.get(
                        "input_tokens", 999999999
                    )
                    row["output_cost"] = model_info.get("output_cost", -1) * row.get(
                        "output_tokens", 999999999
                    )
                    writer.writerow(row)
        except (OSError, IOError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to write executions to CSV: {e}")

    def clear_executions(self) -> None:
        """
        Clears all registered executions.
        """
        self._executions.clear()


if __name__ == "__main__":
    estimator = CostEstimator()
    estimator.register_model("ModelA", {"input_cost": 0.1, "output_cost": 0.5})
    estimator.register_model("ModelB", {"input_cost": 0.025, "output_cost": 0.15})
    estimator.register_execution(
        "detection",
        {
            "model": "ModelA",
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 0,
        },
    )
    estimator.register_execution(
        "classification",
        {
            "model": "ModelB",
            "input_tokens": 200,
            "output_tokens": 100,
            "cached_tokens": 0,
        },
    )

    print(estimator.estimate_cost())
    estimator.export_executions_to_csv("test_executions.csv")
