from concurrent.futures import ProcessPoolExecutor
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.interaction.evaluate_interaction import worker_evaluate_home


def process_home(
    home_index: int,
    qdrant_url: str,
    database_path: str,
    output_dir: str,
    fractions: list[float],
    home_ids: list[int],
):
    client = QdrantClient(url=qdrant_url)
    home_id = home_ids[home_index]
    home_name = f"Home{home_id:02d}"
    collection_name = f"House_{home_id:02d}_online_AD"
    home_output_dir = os.path.join(output_dir, home_name)
    os.makedirs(home_output_dir, exist_ok=True)

    print(f"Processing {home_name}...")

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=99999,
        with_payload=True,
        with_vectors=False,
    )

    points_per_fraction = [int(len(points) * frac) for frac in fractions]

    # with open(
    #     os.path.join(home_output_dir, "all_knowledge.json"), "w", encoding="utf-8"
    # ) as f:
    #     knowledge = {}

    #     for point in points:
    #         knowledge[point.id] = point.payload

    #     json.dump(knowledge, f, indent=4)

    for fraction, points_count in zip(fractions, points_per_fraction):
        print(f"Selecting {points_count} points out of {len(points)}")
        total_to_remove = len(points) - points_count
        points_to_remove = random.sample(points, total_to_remove)

        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[p.id for p in points_to_remove],
            ),
        )

        fraction_output_dir = os.path.join(output_dir, f"{fraction:.2f}")
        worker_evaluate_home(
            home_id=home_id,
            dataset_base_path=database_path,
            output_base_dir=fraction_output_dir,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            model_id="openai/gpt-oss-120b",
            temperature=0.0,
            top_p=0.1,
            max_tokens=64000,
            timeout=60.0,
            use_additional_knowledge=True,
            allowed_question_types=[
                "adversarial_questions",
                "basic_questions",
                "follow_up_questions",
                "indirect_questions",
            ],
        )

        points, _ = client.scroll(
            collection_name=collection_name,
            limit=99999,
            with_payload=True,
            with_vectors=False,
        )


if __name__ == "__main__":
    load_dotenv()
    HOME_IDS = list(range(1, 31))
    HOME_IDS = random.sample(HOME_IDS, 5)
    FRACTIONS = [0.75, 0.5, 0.25]

    DATABASE_PATH: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_DIR = rf"{DATABASE_PATH}\results\ablation_study\knowledge_sensitivity"

    QDRANT_URL = "http://localhost:6333"

    with ProcessPoolExecutor(max_workers=60) as executor:
        futures = [
            executor.submit(
                process_home,
                home_index,
                QDRANT_URL,
                DATABASE_PATH,
                OUTPUT_DIR,
                FRACTIONS,
                HOME_IDS,
            )
            for home_index in range(len(HOME_IDS))
        ]
        for future in futures:
            future.result()
