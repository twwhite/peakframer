"""Diversity sampling via k-means clustering on frame embeddings."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from peakframer.utils import get_logger

logger = get_logger(__name__)


def select_diverse_indices(
    embeddings: np.ndarray,
    count: int,
    random_state: int = 42,
    oversample_factor: int = 3,
) -> list[int]:
    n = len(embeddings)
    if count > n:
        raise ValueError(
            f"Requested {count} frames but only {n} candidate frames available."
        )
    if count == n:
        return list(range(n))

    n_clusters = min(count * oversample_factor, n)
    logger.info(f"Clustering {n} embeddings into {count} groups...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        batch_size=min(4096, n),
    )
    kmeans.fit(embeddings)

    centroids = kmeans.cluster_centers_
    candidate_indices = []
    for centroid in centroids:
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        candidate_indices.append(int(np.argmin(dists)))

    selected = [candidate_indices[0]]
    remaining = candidate_indices[1:]
    while len(selected) < count and remaining:
        best = max(
            remaining,
            key=lambda i: min(
                np.linalg.norm(embeddings[i] - embeddings[s]) for s in selected
            ),
        )
        selected.append(best)
        remaining.remove(best)

    return selected


def compute_diversity_score(
    embeddings: np.ndarray,
    indices: list[int],
) -> float:
    selected = embeddings[indices]
    dot_products = selected @ selected.T
    n = len(selected)
    # mean pairwise cosine distance, excluding self-comparisons
    mask = ~np.eye(n, dtype=bool)
    return float(1 - dot_products[mask].mean())


def compute_random_baseline(
    embeddings: np.ndarray,
    count: int,
    n_trials: int = 10,
    random_state: int = 42,
) -> float:
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_trials):
        indices = list(map(int, rng.choice(len(embeddings), size=count, replace=False)))
        scores.append(compute_diversity_score(embeddings, indices))
    return float(np.mean(scores))
