"""Vector store for file embeddings using ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a vector search query."""

    file_id: str
    score: float  # Cosine similarity (0-1, higher is better)
    metadata: Dict[str, Any]


class VectorStore:
    """ChromaDB-based vector store for file embeddings.

    Manages embeddings and metadata for library files with automatic
    persistence and simple CRUD operations.
    """

    _client: chromadb.ClientAPI
    _collection: chromadb.Collection
    _collection_name: str
    _storage_path: Path

    def __init__(
        self,
        storage_path: Path | str,
        collection_name: str = "library_files",
    ) -> None:
        """Initialize the vector store.

        Args:
            storage_path: Directory to store the ChromaDB database
            collection_name: Name of the collection to use
        """
        self._storage_path = Path(storage_path)
        self._collection_name = collection_name

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initializing ChromaDB at {self._storage_path}")

        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self._storage_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance
        )

        logger.debug(
            f"Vector store ready with {self._collection.count()} embeddings"
        )

    def add(
        self,
        file_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """Add or update a file embedding.

        Args:
            file_id: Unique identifier for the file
            embedding: Normalized embedding vector
            metadata: Metadata to store (e.g., path, hash, description)
        """
        if embedding.ndim != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

        # ChromaDB expects List[float]
        embedding_list = embedding.tolist()

        # Add to collection (upsert behavior)
        self._collection.upsert(
            ids=[file_id],
            embeddings=[embedding_list],
            metadatas=[metadata],
        )

        logger.debug(f"Added embedding for file_id={file_id}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar files.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filter (e.g., {"author": "Smith"})

        Returns:
            List of search results ordered by similarity (descending)
        """
        if query_embedding.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1D, got shape {query_embedding.shape}"
            )

        # Check if collection is empty
        if self._collection.count() == 0:
            logger.warning("Vector store is empty, returning no results")
            return []

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            where=where,
        )

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for file_id, distance, metadata in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                # Convert distance to similarity score
                # ChromaDB uses cosine distance: distance = 1 - cosine_similarity
                # So similarity = 1 - distance
                similarity = 1.0 - distance

                search_results.append(
                    SearchResult(
                        file_id=file_id,
                        score=similarity,
                        metadata=metadata or {},
                    )
                )

        logger.debug(f"Found {len(search_results)} results for query")
        return search_results

    def delete(self, file_id: str) -> None:
        """Delete a file's embedding."""
        
        try:
            self._collection.delete(ids=[file_id])
            logger.debug(f"Deleted embedding for file_id={file_id}")
        except Exception as exc:
            logger.warning(f"Failed to delete file_id={file_id}: {exc}")

    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""

        try:
            results = self._collection.get(ids=[file_id])
            if results["ids"] and results["metadatas"]:
                return results["metadatas"][0]
            return None
        except Exception as exc:
            logger.warning(f"Failed to get file_id={file_id}: {exc}")
            return None

    def update_metadata(
        self, file_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Update metadata for an existing file without changing embedding."""

        # Get existing data
        existing = self._collection.get(ids=[file_id], include=["embeddings", "metadatas"])

        if not existing["ids"]:
            raise ValueError(f"File not found: {file_id}")

        # Merge metadata
        old_metadata = existing["metadatas"][0] or {}
        new_metadata = {**old_metadata, **metadata}

        # Update with same embedding
        self._collection.update(
            ids=[file_id],
            embeddings=existing["embeddings"][0],
            metadatas=[new_metadata],
        )

        logger.debug(f"Updated metadata for file_id={file_id}")

    def count(self) -> int:
        """Get total number of embeddings in the store."""

        return self._collection.count()

    def reset(self) -> None:
        """Delete all embeddings in the collection."""
        
        logger.warning(f"Resetting collection '{self._collection_name}'")
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def __repr__(self) -> str:
        return f"VectorStore(path={self._storage_path}, count={self.count()})"
