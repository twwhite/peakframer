"""Diversity sampling via k-means clustering on frame embeddings."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from peakframer.utils import get_logger

logger = get_logger(__name__)


def select_diverse_indices(
    embeddings: np.ndarray,
    count: int,
    random_state: int = 42,
) -> list[int]:
    n = len(embeddings)
    if count > n:
        raise ValueError(
            f"Requested {count} frames but only {n} candidate frames available."
        )
    if count == n:
        return list(range(n))

    logger.info(f"Clustering {n} embeddings into {count} groups...")

    kmeans = MiniBatchKMeans(
        n_clusters=count,
        random_state=random_state,
        n_init="auto",
        batch_size=min(4096, n),
    )
    kmeans.fit(embeddings)

    centroids = kmeans.cluster_centers_
    selected = []
    for centroid in centroids:
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        selected.append(int(np.argmin(dists)))

    return list(dict.fromkeys(selected))
