"""SigLIP 2 vision backend — image/text embeddings via HuggingFace transformers.

Uses Google SigLIP 2 for joint image-text embeddings with L2 normalization.
Requires: transformers, torch, Pillow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ave._compat import import_optional
from ave.tools.vision import VisionError

if TYPE_CHECKING:
    import numpy as np


class SigLIP2Backend:
    """Vision embedding backend using Google SigLIP 2.

    Uses HuggingFace transformers for model loading and inference.
    All inference runs under torch.no_grad() for efficiency.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise VisionError(
                "Missing optional dependency 'transformers'. Install with: pip install ave[vision]"
            ) from None

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def embed_image(self, image: np.ndarray) -> list[float]:
        """Embed a single image (H, W, 3 uint8) into a normalized vector.

        Args:
            image: RGB image as numpy array with shape (H, W, 3) and dtype uint8.

        Returns:
            L2-normalized embedding as list of floats.
        """
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        np = import_optional("numpy")
        embedding = outputs[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string into a normalized vector.

        Args:
            text: Text string to embed.

        Returns:
            L2-normalized embedding as list of floats.
        """
        import torch

        np = import_optional("numpy")
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

        embedding = outputs[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def embed_batch(self, images: list[np.ndarray]) -> list[list[float]]:
        """Embed multiple images in one forward pass.

        Args:
            images: List of RGB images as numpy arrays (H, W, 3) uint8.

        Returns:
            List of L2-normalized embeddings, one per image.
        """
        import torch
        from PIL import Image

        np = import_optional("numpy")
        pil_images = [Image.fromarray(img) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        embeddings = outputs.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        return embeddings.tolist()
