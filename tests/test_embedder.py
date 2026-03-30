from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from peakframer.embedder import CLIPEmbedder


@patch("peakframer.embedder.open_clip.create_model_and_transforms")
def test_embed_returns_correct_shape(mock_create: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.randn(3, 512)
    mock_model.to.return_value = mock_model  # .to(device) returns itself
    mock_model.eval.return_value = mock_model  # .eval() returns itself
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    mock_create.return_value = (mock_model, None, mock_preprocess)

    embedder = CLIPEmbedder(device=torch.device("cpu"), batch_size=8)
    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
    result = embedder.embed(images)

    assert result.ndim == 2
    assert result.shape == (3, 512)
    assert result.dtype == np.float32


@patch("peakframer.embedder.open_clip.create_model_and_transforms")
def test_embed_l2_normalised(mock_create: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.tensor([[3.0, 4.0]])
    mock_model.to.return_value = mock_model  # .to(device) returns itself
    mock_model.eval.return_value = mock_model  # .eval() returns itself
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    mock_create.return_value = (mock_model, None, mock_preprocess)

    embedder = CLIPEmbedder(device=torch.device("cpu"))
    result = embedder.embed([np.zeros((224, 224, 3), dtype=np.uint8)])

    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@patch("peakframer.embedder.open_clip.create_model_and_transforms")
def test_embed_batching(mock_create: MagicMock) -> None:
    n, d, batch = 10, 512, 3
    mock_model = MagicMock()
    mock_model.encode_image.side_effect = [
        torch.randn(min(batch, n - i * batch), d)
        for i in range((n + batch - 1) // batch)
    ]
    mock_model.to.return_value = mock_model  # .to(device) returns itself
    mock_model.eval.return_value = mock_model  # .eval() returns itself
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    mock_create.return_value = (mock_model, None, mock_preprocess)

    embedder = CLIPEmbedder(device=torch.device("cpu"), batch_size=batch)
    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(n)]
    result = embedder.embed(images)

    assert result.shape[0] == n


@patch("peakframer.embedder.open_clip.create_model_and_transforms")
def test_embed_empty_raises(mock_create: MagicMock) -> None:
    mock_create.return_value = (MagicMock(), None, MagicMock())

    embedder = CLIPEmbedder(device=torch.device("cpu"))
    with pytest.raises(Exception):
        embedder.embed([])
