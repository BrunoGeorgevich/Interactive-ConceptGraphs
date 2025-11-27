from agno.models.openrouter import OpenRouter
from agno.models.lmstudio import LMStudio
from typing import Dict, Any, Optional
from agno.agent import Agent
import json
import os

from conceptgraph.interaction.prompts import (
    AGENT_PROMPT_V3,
    INTENTION_INTERPRETATION_PROMPT,
)
from conceptgraph.interaction.schemas import SystemConfig


class AgentOrchestrator:
    """
    Manages LLM agents for intention interpretation and response generation.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Initializes the agent orchestrator.

        :param config: System configuration.
        :type config: SystemConfig
        """
        self.config = config
        self.interpreter_agent: Optional[Agent] = None
        self.bot_agent: Optional[Agent] = None
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """
        Sets up the Agno agents based on configuration (online/offline).

        """

        if self.config.prefix == "offline":
            interpreter_model = LMStudio(
                id=self.config.local_model_id,
                temperature=0.0,
                top_p=0.1,
                reasoning_effort="high",
                max_tokens=16000,
            )
        else:
            interpreter_model = OpenRouter(
                id=self.config.remote_model_id,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.0,
                top_p=0.1,
                reasoning_effort="high",
                max_tokens=16000,
            )

        self.interpreter_agent = Agent(
            model=interpreter_model,
            instructions=INTENTION_INTERPRETATION_PROMPT,
            markdown=True,
            description="Intent Interpreter",
        )

        if self.config.prefix == "offline":
            bot_model = LMStudio(id=self.config.local_model_id)
        else:
            bot_model = OpenRouter(
                id=self.config.remote_model_id, api_key=os.getenv("OPENROUTER_API_KEY")
            )

        self.bot_agent = Agent(
            model=bot_model,
            markdown=True,
            description="Smart Wheelchair Navigator",
            instructions=AGENT_PROMPT_V3,
        )

    def interpret_intent(self, input_prompt: str) -> Dict[str, Any]:
        """
        Runs the interpreter agent to analyze user intent.

        :param input_prompt: The formulated prompt string for the interpreter.
        :type input_prompt: str
        :return: Parsed JSON response.
        :rtype: Dict[str, Any]
        """
        try:
            response = self.interpreter_agent.run(input_prompt).content

            clean_response = response.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(clean_response)
                return result
            except json.JSONDecodeError:
                return {
                    "state": "UNCLEAR",
                    "direct_response": "I couldn't process that request properly.",
                    "raw_response": response,
                }
        except Exception as e:
            return {"state": "ERROR", "direct_response": f"An error occurred: {str(e)}"}

    def generate_bot_response(self, input_prompt: str) -> str:
        """
        Runs the bot agent to generate the final response.

        :param input_prompt: The formulated prompt string for the bot.
        :type input_prompt: str
        :return: Raw string response from the bot.
        :rtype: str
        """
        try:
            response = self.bot_agent.run(input_prompt)
            return response.content
        except Exception as e:
            return f"<message>Error generating response: {str(e)}</message>"
