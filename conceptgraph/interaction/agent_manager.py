from typing import Dict, Any
from openai import OpenAI
import traceback
import json
import os

from conceptgraph.interaction.prompts import (
    AGENT_PROMPT_V3,
    INTENTION_INTERPRETATION_PROMPT,
)
from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.inference.cost_estimator import CostEstimator


class AgentOrchestrator:
    """
    Manages LLM agents for intention interpretation and response generation using the OpenAI framework.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Initializes the agent orchestrator with OpenAI clients and prompt templates.

        :param config: System configuration.
        :type config: SystemConfig
        """
        self.config = config
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """
        Sets up the OpenAI client based on configuration (online/offline).

        :raises RuntimeError: If API key is missing in online mode.
        :return: None
        :rtype: None
        """
        if self.config.prefix == "offline":
            self.openai_client = OpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                timeout=60.0,
            )
            self.model_id = self.config.local_model_id
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=60.0,
            )
            self.model_id = self.config.remote_model_id

    def _register_cost(self, response: Any, execution_type: str) -> None:
        """
        Registers execution cost with CostEstimator if usage data is available.

        :param response: Response object from OpenAI API.
        :type response: Any
        :param execution_type: Type of execution for cost tracking.
        :type execution_type: str
        :return: None
        :rtype: None
        """
        try:
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                execution_data = {
                    "model": self.model_id,
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "cached_tokens": (
                        getattr(
                            getattr(usage, "prompt_tokens_details", {}),
                            "cached_tokens",
                            0,
                        )
                        if hasattr(usage, "prompt_tokens_details")
                        else 0
                    ),
                }
                CostEstimator().register_execution(execution_type, execution_data)
        except (AttributeError, TypeError, ValueError):
            traceback.print_exc()

    def interpret_intent(self, input_prompt: str) -> Dict[str, Any]:
        """
        Analyzes user intent using the LLM.

        :param input_prompt: The formulated prompt string for the interpreter.
        :type input_prompt: str
        :raises RuntimeError: If the LLM call fails after retries.
        :return: Parsed JSON response.
        :rtype: Dict[str, Any]
        """
        max_retries: int = 3
        for attempt in range(max_retries):
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": INTENTION_INTERPRETATION_PROMPT,
                        },
                        {"role": "user", "content": input_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.1,
                    max_tokens=64000,
                )
                self._register_cost(completion, "intent_interpretation")
                response = completion.choices[0].message
                content = getattr(response, "content", "")
                clean_response = (
                    content.replace("```json", "").replace("```", "").strip()
                )
                try:
                    result = json.loads(clean_response)
                    result["raw_response"] = response.model_dump()
                    return result
                except json.JSONDecodeError:
                    return {
                        "state": "UNCLEAR",
                        "direct_response": "I couldn't process that request properly.",
                        "raw_response": response.model_dump(),
                    }
            except (OSError, AttributeError, TypeError, ValueError) as e:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM agent call failed after retries: {e}")
        return {
            "state": "ERROR",
            "direct_response": "An error occurred during intent interpretation.",
        }

    def generate_bot_response(self, input_prompt: str) -> Dict[str, str]:
        """
        Generates the final response using the LLM.

        :param input_prompt: The formulated prompt string for the bot.
        :type input_prompt: str
        :raises RuntimeError: If the LLM call fails after retries.
        :return: Dictionary containing content and reasoning from the bot.
        :rtype: Dict[str, str]
        """
        max_retries: int = 3
        for attempt in range(max_retries):
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": AGENT_PROMPT_V3},
                        {"role": "user", "content": input_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.1,
                    max_tokens=16000,
                )
                self._register_cost(completion, "response_generation")
                response = completion.choices[0].message
                content = getattr(response, "content", "")
                return {
                    "content": content,
                    "reasoning": getattr(
                        response, "reasoning", "Failed to retrieve reasoning."
                    ),
                }
            except (OSError, AttributeError, TypeError, ValueError) as e:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    return {
                        "content": f"<message>Error generating response: {str(e)}</message>",
                        "reasoning": "Error occurred during generation.",
                    }
        return {
            "content": "<message>Unknown error generating response.</message>",
            "reasoning": "Unknown error occurred.",
        }
