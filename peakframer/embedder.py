"""CLIP-based image embedding — encodes frames into L2-normalised feature vectors."""

import numpy as np
import open_clip
import torch
from PIL import Image

from peakframer.utils import get_logger

logger = get_logger(__name__)


class CLIPEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: torch.device | None = None,
        batch_size: int = 64,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size

        logger.info(f"Loading CLIP {model_name} ({pretrained}) on {self.device}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        logger.info("CLIP model ready")

    @torch.no_grad()
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            tensors = torch.stack(
                [self.preprocess(Image.fromarray(img[..., ::-1])) for img in batch]
            ).to(self.device)

            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)
