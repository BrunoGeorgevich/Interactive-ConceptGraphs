from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """
    Configuration object for the Smart Wheelchair System.
    """

    house_id: int
    dataset_base_path: str
    prefix: str = "online"
    qdrant_url: str = "http://localhost:6333"
    local_data_dir: str = "data"
    force_recreate_table: bool = False
    use_additional_knowledge: bool = True

    remote_model_id: str = "openai/gpt-oss-120b:nitro"
    local_model_id: str = "openai/gpt-oss-20b"

    map_binary_threshold: int = 250
    min_contour_area: int = 100
    crop_padding: int = 5
    trajectory_image_dimming: int = 50

    debug_input_path: str = "data/input_debug.txt"
    debug_output_path: str = "data/output_debug.txt"


@dataclass
class InteractionRequest:
    """
    Encapsulates a user interaction request.
    """

    query: str
    user_pose: Tuple[float, float, float]
    timestamp: float


@dataclass
class InteractionResponse:
    """
    Encapsulates the system's response to an interaction.
    """

    text_response: str
    target_coordinates: Optional[Tuple[float, float, float]] = None
    navigated_room: Optional[str] = None
    selected_object_tag: Optional[str] = None
    intent_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
