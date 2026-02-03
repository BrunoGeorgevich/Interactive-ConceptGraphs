from concurrent.futures import ProcessPoolExecutor
from qdrant_client import QdrantClient
from functools import partial
from tqdm import tqdm


def recover_collection_snapshot(collection_name: str, qdrant_url: str) -> str:
    """
    Recover the last snapshot for a given Qdrant collection.

    :param collection_name: Name of the collection to recover
    :type collection_name: str
    :param qdrant_url: URL of the Qdrant server
    :type qdrant_url: str
    :return: Name of the recovered collection
    :rtype: str
    :raises IndexError: If no snapshots are available for the collection
    :raises ValueError: If collection name or URL is invalid
    """
    client = QdrantClient(url=qdrant_url)
    snapshots = client.list_snapshots(collection_name=collection_name)

    if not snapshots:
        raise IndexError(f"No snapshots available for collection: {collection_name}")

    last_snapshot = snapshots[-1]
    last_snapshot_url = (
        f"{qdrant_url}/collections/{collection_name}/snapshots/{last_snapshot.name}"
    )
    client.recover_snapshot(
        collection_name=collection_name,
        location=last_snapshot_url,
    )
    return collection_name


if __name__ == "__main__":
    COLLECTIONS = [f"House_{i:02d}_online" for i in range(1, 31)]
    COLLECTIONS_WITH_SUFFIX = [f"House_{i:02d}_online_AD" for i in range(1, 31)]
    COLLECTIONS = COLLECTIONS + COLLECTIONS_WITH_SUFFIX
    QDRANT_URL = "http://localhost:6333"

    recover_with_url = partial(recover_collection_snapshot, qdrant_url=QDRANT_URL)

    with ProcessPoolExecutor() as executor:
        futures = executor.map(recover_with_url, COLLECTIONS)
        with tqdm(total=len(COLLECTIONS), desc="Recovering snapshots") as pbar:
            for collection_name in futures:
                pbar.set_postfix_str(f"Restored: {collection_name}")
                pbar.update(1)
