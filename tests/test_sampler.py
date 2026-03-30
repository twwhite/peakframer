import numpy as np
import pytest

from peakframer.sampler import select_diverse_indices


def _random_embeddings(n: int, d: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, d)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_returns_correct_count() -> None:
    embs = _random_embeddings(200)
    result = select_diverse_indices(embs, count=50)
    assert len(result) == 50


def test_returns_valid_indices() -> None:
    embs = _random_embeddings(100)
    result = select_diverse_indices(embs, count=20)
    assert all(0 <= i < 100 for i in result)


def test_no_duplicate_indices() -> None:
    embs = _random_embeddings(100)
    result = select_diverse_indices(embs, count=20)
    assert len(result) == len(set(result))


def test_count_equals_n_returns_all() -> None:
    embs = _random_embeddings(10)
    result = select_diverse_indices(embs, count=10)
    assert sorted(result) == list(range(10))


def test_count_exceeds_n_raises() -> None:
    embs = _random_embeddings(5)
    with pytest.raises(ValueError, match="only 5 candidate frames"):
        select_diverse_indices(embs, count=10)


def test_reproducible_with_seed() -> None:
    embs = _random_embeddings(200)
    r1 = select_diverse_indices(embs, count=30, random_state=42)
    r2 = select_diverse_indices(embs, count=30, random_state=42)
    assert r1 == r2
