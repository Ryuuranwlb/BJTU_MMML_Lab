"""Embedding generation and vector storage for library files."""

from .encoder import OpenCLIPEncoder
from .vector_store import SearchResult, VectorStore

__all__ = ["OpenCLIPEncoder", "VectorStore", "SearchResult"]
