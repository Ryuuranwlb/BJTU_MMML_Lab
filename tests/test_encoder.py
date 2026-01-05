"""Basic tests for OpenCLIP encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pangu_agent.library.embeddings.encoder import OpenCLIPEncoder


@pytest.fixture
def encoder():
    """Create a test encoder instance."""
    return OpenCLIPEncoder(model_name="ViT-B-32", pretrained="openai")


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample image file."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    img.save(img_path)
    return img_path


def test_encoder_initialization(encoder):
    """Test that encoder initializes correctly."""
    assert encoder is not None
    assert encoder.embedding_dim > 0


def test_encode_text(encoder):
    """Test text encoding produces valid embedding."""
    text = "This is a test document about machine learning."
    embedding = encoder.encode_text(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (encoder.embedding_dim,)
    # Check normalization
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_encode_image(encoder, sample_image):
    """Test image encoding produces valid embedding."""
    embedding = encoder.encode_image(sample_image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (encoder.embedding_dim,)
    # Check normalization
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_encode_file(encoder, sample_image):
    """Test generic file encoding interface."""
    embedding = encoder.encode_file(sample_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (encoder.embedding_dim,)
