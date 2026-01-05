"""OpenCLIP-based encoder for generating file embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class OpenCLIPEncoder:
    """Generate embeddings for files using OpenCLIP multimodal models.

    Supports both text (PDF) and image files. Uses CLIP's vision-language
    alignment to produce 512-dimensional embeddings suitable for semantic search.
    """

    _model: torch.nn.Module
    _preprocess: callable
    _tokenizer: callable
    _device: torch.device
    _model_name: str
    _pretrained: str

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the encoder with a specific CLIP model.

        Args:
            model_name: OpenCLIP model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained weights to use (e.g., 'openai', 'laion2b_s34b_b79k')
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self._model_name = model_name
        self._pretrained = pretrained

        # Determine device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.debug(f"Loading OpenCLIP model '{model_name}' with '{pretrained}' weights on {self._device}")

        # Load model
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load OpenCLIP model: {exc}") from exc

        logger.debug(f"OpenCLIP encoder initialized successfully")

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of generated embeddings."""
        # Most CLIP models use 512 dimensions, but ViT-L uses 768
        return self._model.text_projection.shape[1]

    @torch.no_grad()
    def encode_text(self, text: str, max_length: int = 77) -> np.ndarray:
        """Encode text into an embedding vector.

        Args:
            text: Input text to encode
            max_length: Maximum token length (CLIP default is 77)

        Returns:
            Normalized embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Truncate if necessary (CLIP has token limit)
        tokens = self._tokenizer([text]).to(self._device)

        # Generate embedding
        text_features = self._model.encode_text(tokens)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)[0]

    @torch.no_grad()
    def encode_image(self, image_path: Path) -> np.ndarray:
        """Encode an image file into an embedding vector."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

            # Generate embedding
            image_features = self._model.encode_image(image_tensor)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().astype(np.float32)[0]
        except Exception as exc:
            raise RuntimeError(f"Failed to encode image {image_path}: {exc}") from exc

    def encode_pdf(self, pdf_path: Path, max_chars: int = 10000) -> np.ndarray:
        """Encode a PDF file by extracting text and encoding it."""

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))

            # Extract text from all pages
            text_chunks = []
            total_chars = 0

            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_chunks.append(page_text)
                        total_chars += len(page_text)

                        # Stop if we've extracted enough
                        if total_chars >= max_chars:
                            break
                except Exception as exc:
                    logger.warning(f"Failed to extract text from page: {exc}")
                    continue

            full_text = "\n".join(text_chunks)

            if not full_text.strip():
                raise ValueError(f"PDF contains no extractable text: {pdf_path}")

            # Truncate to max_chars
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars]

            return self.encode_text(full_text)

        except Exception as exc:
            raise RuntimeError(f"Failed to encode PDF {pdf_path}: {exc}") from exc

    def encode_file(self, file_path: Path) -> np.ndarray:
        """Encode any supported file type (PDF or image)."""
        
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.encode_pdf(file_path)
        elif suffix in [".jpg", ".jpeg", ".png"]:
            return self.encode_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def __repr__(self) -> str:
        return f"OpenCLIPEncoder(model={self._model_name}, pretrained={self._pretrained}, device={self._device})"
