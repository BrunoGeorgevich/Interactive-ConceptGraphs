from concurrent.futures import ProcessPoolExecutor
from qdrant_client import QdrantClient
from functools import partial
from tqdm import tqdm

import traceback


def create_collection_snapshot(collection_name: str, qdrant_url: str) -> str:
    """
    Create a snapshot for a given Qdrant collection.

    :param collection_name: Name of the collection to snapshot
    :type collection_name: str
    :param qdrant_url: URL of the Qdrant server
    :type qdrant_url: str
    :return: Name of the collection that was snapshotted
    :rtype: str
    :raises RuntimeError: If snapshot creation fails
    """
    try:
        client = QdrantClient(url=qdrant_url)
        client.create_snapshot(collection_name=collection_name)
        return collection_name
    except (ConnectionError, TimeoutError, OSError) as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Failed to create snapshot for collection {collection_name}: {str(e)}"
        )


if __name__ == "__main__":
    COLLECTIONS = [f"House_{i:02d}_online" for i in range(1, 31)]
    COLLECTIONS_WITH_SUFFIX = [f"House_{i:02d}_online_AD" for i in range(1, 31)]
    COLLECTIONS = COLLECTIONS + COLLECTIONS_WITH_SUFFIX
    QDRANT_URL = "http://localhost:6333"

    create_with_url = partial(create_collection_snapshot, qdrant_url=QDRANT_URL)

    with ProcessPoolExecutor() as executor:
        futures = executor.map(create_with_url, COLLECTIONS)
        with tqdm(total=len(COLLECTIONS), desc="Creating snapshots") as pbar:
            for collection_name in futures:
                pbar.set_postfix_str(f"Snapshotted: {collection_name}")
                pbar.update(1)
