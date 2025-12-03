from typing import Optional, List, Dict, Tuple
import numpy as np
import pickle
import time
import math
import json
import gzip
import os

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from conceptgraph.interaction.schemas import (
    SystemConfig,
    InteractionRequest,
    InteractionResponse,
)
from conceptgraph.interaction.spatial_manager import (
    SpatialContextManager,
    SceneGraphManager,
)
from conceptgraph.interaction.memory_engine import SemanticMemoryEngine
from conceptgraph.interaction.agent_manager import AgentOrchestrator
from conceptgraph.interaction.utils import log_debug_interaction
from conceptgraph.interaction.visualizer import MapNavigator
from conceptgraph.interaction.prompts import (
    INTENTION_INTERPRETATION_PROMPT,
    AGENT_PROMPT_V3,
)


class SmartWheelchairSystem:
    """
    Main controller for the Smart Wheelchair Navigation and Interaction System.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Initializes the system components.

        :param config: System configuration object.
        :type config: SystemConfig
        """
        self.config = config
        self.console = Console()

        self.chat_history: List[Dict[str, str]] = []
        self.active_rag_context: List[dict] = []
        self.last_bot_message: str = ""

        self.spatial_manager = SpatialContextManager(config)
        self.memory_engine = SemanticMemoryEngine(config)
        self.agent_manager = AgentOrchestrator(config)
        self.visualizer: Optional[MapNavigator] = None

        self._initialize_memory()

    def _on_map_click(self, new_pos: Tuple[float, float, float]) -> None:
        """
        Callback triggered when the map is clicked. Prints update to console.

        :param new_pos: The new coordinates (x, y, z).
        :type new_pos: Tuple[float, float, float]
        """
        room_name = self.spatial_manager.get_room_name_at_location(new_pos)
        self.console.print(
            f"\n[bold yellow]📍 Location Updated via Map:[/bold yellow] "
            f"Room: [cyan]{room_name}[/cyan] | "
            f"Coords: ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
            "\n Query (or 'q' to quit): ",
            end="",
        )

    def additional_object_converter(self, add_obj: dict) -> dict:
        """
        Converts an additional knowledge object from Qdrant format to the enriched object format.

        :param add_obj: Dictionary containing the 'meta_data' field from Qdrant Record payload.
        :type add_obj: dict
        :raises KeyError: If required keys are missing in the input dictionary.
        :raises TypeError: If input is not a dictionary or coordinates are not a list.
        :return: Dictionary in the enriched object format compatible with the system.
        :rtype: dict
        """

        try:
            if not isinstance(add_obj, dict):
                raise TypeError(
                    "Input must be a dictionary representing Qdrant meta_data."
                )

            meta = add_obj
            required_keys = [
                "type",
                "class_name",
                "description",
                "room_name",
                "coordinates",
                "id",
            ]
            for key in required_keys:
                if key not in meta:
                    raise KeyError(f"Missing required key '{key}' in meta_data.")

            coordinates = meta.get("coordinates", [0.0, 0.0, 0.0])
            if not isinstance(coordinates, list) or len(coordinates) != 3:
                raise TypeError("Coordinates must be a list of three float values.")

            enriched_obj: dict = {
                "id": meta.get("id", ""),
                "class_name": meta.get("class_name", ""),
                "consolidated_caption": meta.get("description", ""),
                "room_name": meta.get("room_name", ""),
                "pcd_np": np.array([coordinates]),
            }
            return enriched_obj

        except (KeyError, TypeError) as err:
            traceback.print_exc()
            raise RuntimeError(f"Error converting additional object: {err}")

    def _initialize_memory(self) -> None:
        """
        Loads object data and ensures Qdrant is populated.

        """
        objects_path = os.path.join(
            self.config.dataset_base_path,
            "outputs",
            f"Home{self.config.house_id:02d}",
            "Wandering",
            "exps",
            f"{self.config.prefix}_house_{self.config.house_id}_map",
            f"pcd_{self.config.prefix}_house_{self.config.house_id}_map.pkl.gz",
        )

        if os.path.exists(objects_path):
            with gzip.open(objects_path, "rb") as f:
                raw_results = pickle.load(f)
            raw_objects = raw_results["objects"]

            self.debug_map_img, enriched_objects = (
                self.spatial_manager.inject_objects_and_build_graph(raw_objects)
            )

            self.memory_engine.ensure_collection_ready(
                self.memory_engine.vector_db, enriched_objects
            )
            self.memory_engine.ensure_collection_ready(
                self.memory_engine.additional_knowledge_db
            )

            if self.config.use_additional_knowledge:
                additional_knowledge_elements = (
                    self.memory_engine.additional_knowledge_db.client.scroll(
                        self.memory_engine.additional_knowledge_db.collection,
                        limit=9999999999,
                    )[0]
                )
                additional_knowledge_objects = list(
                    filter(
                        lambda x: x.payload["meta_data"]["type"] == "object",
                        additional_knowledge_elements,
                    )
                )

                additional_knowledge_objects = list(
                    map(lambda x: x.payload["meta_data"], additional_knowledge_objects)
                )

                additional_knowledge_objects = list(
                    map(self.additional_object_converter, additional_knowledge_objects)
                )

                enriched_objects.extend(additional_knowledge_objects)

            self.spatial_manager.scene_manager = SceneGraphManager(enriched_objects)
        else:

            pass

    def clear_active_memory(self) -> None:
        """
        Clears the active RAG context memory.

        """
        self.active_rag_context = []
        self.last_bot_message = ""
        self.chat_history = []

    def start_interactive_session(
        self,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        verbose: bool = True,
    ) -> None:
        """
        Starts the interactive console loop with map visualization.

        :param initial_pose: Initial user position.
        :type initial_pose: tuple[float, float, float]
        :param verbose: Whether to print progress to the console.
        :type verbose: bool
        """
        self.console.print(
            Panel(
                f"[bold cyan]Session for House {self.config.house_id}[/bold cyan]",
                title="[bold green]Initialization[/bold green]",
                expand=False,
            )
        )

        self.visualizer = MapNavigator(
            "Interactive Map (Press 'q' to quit)",
            self.spatial_manager.reconstruct_debug_image(),
            self.spatial_manager.map_origin,
            self.spatial_manager.map_resolution,
            initial_pose,
        )

        self.visualizer.set_change_room_callback(self._on_map_click)

        self.visualizer.start()

        self.console.print(
            Panel(
                "[bold green]Starting Chat Session[/bold green]\n"
                "Click on the map window to move the user. Type your query below.",
                title="[bold cyan]Ready[/bold cyan]",
                expand=False,
            )
        )

        while True:
            try:
                if self.visualizer.should_exit():
                    break

                display_pose = self.visualizer.user_pos
                display_room = self.spatial_manager.get_room_name_at_location(
                    display_pose
                )

                user_input = Prompt.ask(
                    f"\n[cyan][Pos: {display_pose[0]:.2f}, {display_pose[1]:.2f}, {display_pose[2]:.2f}] "
                    f"Room: {display_room}[/cyan] Query (or 'q' to quit): "
                )

                if user_input.lower() in ["q", "quit", "exit"]:
                    self.visualizer.stop()
                    break

                actual_pose = self.visualizer.user_pos

                if user_input.lower().startswith("move"):
                    try:
                        parts = [
                            float(x.strip())
                            for x in user_input.lower().replace("move", "").split(",")
                        ]
                        if len(parts) == 3:
                            self.visualizer.user_pos = tuple(parts)
                            continue
                    except ValueError:
                        self.console.print("[yellow]Invalid movement format.[/yellow]")
                        continue

                request = InteractionRequest(
                    query=user_input, user_pose=actual_pose, timestamp=time.time()
                )

                response = self.process_interaction(request, verbose=verbose)

                if response.target_coordinates:
                    self.visualizer.move_to_coordinate(response.target_coordinates)

            except KeyboardInterrupt:
                break

        if self.visualizer:
            self.visualizer.stop()
            self.visualizer.join()

    def process_query(
        self,
        query: str,
        user_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> InteractionResponse:
        """
        Processes a single query without interactive console or visualization.

        :param query: The user query to process.
        :type query: str
        :param user_pose: The current user position coordinates.
        :type user_pose: tuple[float, float, float]
        :return: The structured interaction response.
        :rtype: InteractionResponse
        """
        request = InteractionRequest(
            query=query, user_pose=user_pose, timestamp=time.time()
        )

        return self.process_interaction(request, verbose=False)

    def process_interaction(
        self, request: InteractionRequest, verbose: bool = False
    ) -> InteractionResponse:
        """
        Executes the full interaction pipeline.

        :param request: The interaction request data.
        :type request: InteractionRequest
        :param verbose: Whether to print progress to the console.
        :type verbose: bool
        :return: The structured interaction response.
        :rtype: InteractionResponse
        """
        start_time = time.time()

        room_name = self.spatial_manager.get_room_name_at_location(request.user_pose)
        scene_tree = self.spatial_manager.scene_manager.get_text_representation(
            request.user_pose
        )
        history_str = "\n".join(
            [f"State: {h['state']}\nUser: {h['user']}\nBot: {h['bot']}" for h in self.chat_history]
        )

        intention_input = (
            f"<CURRENT_ROOM>{room_name}</CURRENT_ROOM>\n"
            f"<SCENE_GRAPH_SUMMARY>\n{scene_tree}\n</SCENE_GRAPH_SUMMARY>\n\n"
            f"<LAST_BOT_MESSAGE>{self.last_bot_message}</LAST_BOT_MESSAGE>\n\n"
            f"<CHAT_HISTORY>\n{history_str}\n</CHAT_HISTORY>\n\n"
            f"<USER_QUERY>{request.query}</USER_QUERY>"
        )

        log_debug_interaction(
            self.config.debug_input_path,
            "INTERPRETER",
            system_prompt=INTENTION_INTERPRETATION_PROMPT,
            user_input=intention_input,
            mode="w",
        )

        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]Interpreter processing...", total=None)
                intent_data = self.agent_manager.interpret_intent(intention_input)
                console.print(
                    f"[magenta][INTERPRETER] Response:[/magenta]\n{intent_data}"
                )
                progress.remove_task(task)
        else:
            intent_data = self.agent_manager.interpret_intent(intention_input)

        log_debug_interaction(
            self.config.debug_output_path,
            "INTERPRETER",
            content=json.dumps(intent_data, indent=2),
            mode="w",
        )

        state = intent_data.get("state", "NEW_REQUEST")
        direct_response = intent_data.get("direct_response")

        if verbose:
            self.console.print(
                f"\n[bold magenta][INTERPRETER] State:[/bold magenta] [cyan]{state}[/cyan]"
            )

        final_text = direct_response
        target_coords = None
        navigated_room = None
        selected_object_tag = None
        rag_docs = []

        if state == "TAKE_ME_TO_ROOM":
            selected_room = intent_data.get("selected_room", {})
            center_coords = selected_room.get("center_coordinates")
            if center_coords:
                target_coords = tuple(center_coords)
                navigated_room = selected_room.get("room_name")
                if verbose:
                    self.console.print(
                        f"[bold green]Navigating to room:[/bold green] {navigated_room}"
                    )
        elif state == "ADDITIONAL_INFORMATION":
            final_text = intent_data.get("direct_response", "")
            self.memory_engine.add_additional_information(
                intent_data.get("additional_knowledge", []),
            )

        elif state == "ADD_INFO_DESAMBIGUATION":
            final_text = intent_data.get("direct_response", "")

        elif state not in ["END_CONVERSATION", "UNCLEAR", "SCENE_GRAPH_QUERY"]:

            if state == "CONTINUE_REQUEST":
                rag_docs = self.active_rag_context
                if verbose:
                    self.console.print("[blue]Using Active Memory[/blue]")
            else:
                queries = intent_data.get("rag_queries", [])
                if len(queries) > 0:
                    queries.append(request.query)
                rerank_query = intent_data.get("rerank_query", "")
                if queries:
                    if verbose:
                        self.console.print(f"[blue]Queries:[/blue] {queries}")
                    raw_chunks = self.memory_engine.query_relevant_chunks(
                        queries=queries, rerank_query=rerank_query
                    )
                    rag_docs = []
                    for _, meta, _ in raw_chunks:
                        centroid = meta.get("centroid")
                        dist = (
                            math.dist(centroid, request.user_pose) if centroid else None
                        )
                        meta["distance_to_user"] = dist
                        rag_docs.append(meta)

                    if verbose:
                        self.console.print(
                            f"[blue]Retrieved {len(rag_docs)} RAG Documents.[/blue]"
                        )

                    self.active_rag_context = rag_docs

        log_debug_interaction(
            self.config.debug_output_path,
            "RAG",
            rag_docs=rag_docs,
            mode="a",
        )

        should_run_bot = state not in [
            "END_CONVERSATION",
            "UNCLEAR",
            "ADDITIONAL_INFORMATION",
            "ADD_INFO_DESAMBIGUATION",
            "SCENE_GRAPH_QUERY",
            "TAKE_ME_TO_ROOM",
        ]

        if should_run_bot:
            history_str = "\n".join(
                [f"State: {h['state']}\nUser: {h['user']}\nBot: {h['bot']}" for h in self.chat_history]
            )
            rag_context_str = json.dumps(rag_docs, indent=2, ensure_ascii=False)

            bot_input = f"""
<INPUT_DATA>
    <USER_QUERY>{request.query}</USER_QUERY>
    <CURRENT_ROOM>{room_name}</CURRENT_ROOM>
    <CURRENT_STATE>{state}</CURRENT_STATE>
    <CHAT_HISTORY>
    {history_str}
    </CHAT_HISTORY>
    <INTERPRETED_INTENT>
    {intent_data.get('intent_explanation', '')}
    </INTERPRETED_INTENT>
    <RETRIEVED_CONTEXT_RAG>
    {rag_context_str}
    </RETRIEVED_CONTEXT_RAG>
</INPUT_DATA>
"""
            log_debug_interaction(
                self.config.debug_input_path,
                "BOT",
                system_prompt=AGENT_PROMPT_V3,
                user_input=bot_input,
                mode="a",
            )

            if verbose:
                self.console.print("[cyan]Generating Bot Response...[/cyan]")
            bot_raw_response = self.agent_manager.generate_bot_response(bot_input)
            if verbose:
                self.console.print(
                    f"[green][BOT] Response:[/green]\n{bot_raw_response['content']}\nReasoning: {bot_raw_response['reasoning']}"
                )
            log_debug_interaction(
                self.config.debug_output_path, "BOT", content=bot_raw_response, mode="a"
            )

            final_text = bot_raw_response["content"]
            if "<message>" in bot_raw_response["content"]:
                try:
                    final_text = (
                        bot_raw_response["content"]
                        .split("<message>")[1]
                        .split("</message>")[0]
                        .strip()
                    )
                except (IndexError, ValueError, KeyError):
                    traceback.print_exc()

            elif "<answer>" in bot_raw_response["content"]:
                try:
                    final_text = (
                        bot_raw_response["content"]
                        .split("<answer>")[1]
                        .split("</answer>")[0]
                        .strip()
                    )
                except (IndexError, ValueError, KeyError):
                    traceback.print_exc()

            if "<target_coordinates>" in bot_raw_response["content"]:
                try:
                    coords_str = (
                        bot_raw_response["content"]
                        .split("<target_coordinates>")[1]
                        .split("</target_coordinates>")[0]
                        .strip()
                    )
                    target_coords = tuple(
                        [float(x.strip()) for x in coords_str.split(",")]
                    )
                except (IndexError, ValueError, KeyError):
                    traceback.print_exc()

            if "<object_tag>" in bot_raw_response.get("content", ""):
                try:
                    selected_object_tag = (
                        bot_raw_response["content"]
                        .split("<object_tag>")[1]
                        .split("</object_tag>")[0]
                        .strip()
                    )
                except (IndexError, KeyError):
                    traceback.print_exc()

            if "<room_name>" in bot_raw_response["content"]:
                try:
                    navigated_room = (
                        bot_raw_response["content"]
                        .split("<room_name>")[1]
                        .split("</room_name>")[0]
                        .strip()
                    )
                except (IndexError, KeyError):
                    traceback.print_exc()

        self.last_bot_message = final_text or ""
        self.chat_history.append({"state": state, "user": request.query, "bot": final_text or ""})
        if len(self.chat_history) > 5:
            self.chat_history.pop(0)

        if verbose and final_text:
            self.console.print(f"[bold green]Assistant:[/bold green] {final_text}")

        return InteractionResponse(
            text_response=final_text if final_text else "",
            target_coordinates=target_coords,
            navigated_room=navigated_room,
            selected_object_tag=selected_object_tag,
            intent_metadata=intent_data,
            retrieved_context=rag_docs,
            execution_time=time.time() - start_time,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import traceback

    load_dotenv()

    DATASET_BASE_PATH = "D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge"

    config = SystemConfig(
        house_id=1,
        dataset_base_path=DATASET_BASE_PATH,
        prefix="online",
        qdrant_url="http://localhost:6333",
        force_recreate_table=False,
        local_data_dir="data",
        use_additional_knowledge=True,
        debug_input_path=os.path.join("data", "input_debug.txt"),
        debug_output_path=os.path.join("data", "output_debug.txt"),
    )

    console = Console()

    try:
        if not os.path.exists(DATASET_BASE_PATH):
            console.print(
                f"[bold red]Warning: Dataset path '{DATASET_BASE_PATH}' does not exist.[/bold red]"
            )
        console.print("[cyan]Initializing Smart Wheelchair System...[/cyan]")
        system = SmartWheelchairSystem(config)
        system.start_interactive_session(initial_pose=(0.0, 0.0, 0.0))

    except Exception as e:
        console.print(f"[bold red]Fatal Error in Main Loop:[/bold red] {e}")
        traceback.print_exc()
