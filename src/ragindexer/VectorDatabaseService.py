# -*- coding: utf-8 -*-
"""
Vector Database Service Component

Stores and indexes embeddings for semantic search using Qdrant.
Supports in-memory and on-disk persistence.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from ragindexer.EmbeddingService import EmbeddedChunk

logger = logging.getLogger(__name__)


class StoredEmbedding(BaseModel):
    """
    A stored embedding in the vector database.

    Attributes:
        point_id: Unique ID in the vector database
        chunk_content: Original text content
        embedding: Vector representation
        source_file: Source document path
        document_title: Document title if available
        chunk_index: Index of chunk in document
        total_chunks: Total chunks in document
        start_char: Character position in original text
        end_char: Character position in original text
        stored_at: When the embedding was stored
    """

    point_id: str
    chunk_content: str
    embedding: List[float]
    source_file: str
    document_title: Optional[str] = None
    document_author: Optional[str] = None
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    stored_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """
    Result of a vector database search.

    Attributes:
        point_id: ID in the vector database
        chunk_content: Retrieved text content
        score: Similarity score (0-1, higher is more similar)
        source_file: Source document path
        chunk_index: Index in the document
    """

    point_id: str
    chunk_content: str
    score: float
    source_file: str
    document_title: Optional[str] = None
    chunk_index: int


class VectorDatabaseResult(BaseModel):
    """
    Result of vector database operations.

    Attributes:
        operation: Operation performed (add, search, delete)
        success: Whether operation succeeded
        items_affected: Number of items affected
        results: Search results if applicable
        error: Error message if failed
        duration_seconds: Operation duration
        timestamp: When operation occurred
    """

    operation: str
    success: bool
    items_affected: int = 0
    results: List[SearchResult] = []
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class VectorDatabaseService:
    """
    Service for storing and searching embeddings using Qdrant.

    Supports both in-memory and persistent (file-based) storage.
    Enables semantic search through vector similarity.
    """

    def __init__(
        self,
        collection_name: str = "ragindexer_embeddings",
        vector_size: int = 384,
        persistence_path: Optional[Path] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Vector Database Service.

        Args:
            collection_name: Name of the Qdrant collection
            vector_size: Dimension of embedding vectors (384 for MiniLM-L6-v2)
            persistence_path: Path for on-disk storage (None = in-memory)
            logger_instance: Logger to use (defaults to module logger)

        Note:
            If persistence_path is None, uses in-memory storage (data lost on exit).
            If persistence_path is provided, creates SQLite-backed persistence.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.persistence_path = persistence_path
        self.logger = logger_instance or logger

        # Initialize Qdrant client
        if persistence_path is None:
            # In-memory mode
            self.logger.info("Initializing Qdrant in-memory client")
            self.client = QdrantClient(":memory:")
        else:
            # Persistent mode (file-based)
            persistence_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initializing Qdrant persistent client at {persistence_path}")
            self.client = QdrantClient(path=str(persistence_path))

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

        self.logger.info(
            f"Vector Database initialized: collection='{collection_name}', "
            f"vector_size={vector_size}, persistence={persistence_path is not None}"
        )

    def _ensure_collection_exists(self) -> None:
        """
        Create the collection if it doesn't already exist.

        Uses cosine distance for similarity (recommended for embeddings).
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            self.logger.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> VectorDatabaseResult:
        """
        Add embedded chunks to the vector database.

        Args:
            embedded_chunks: List of EmbeddedChunk from EmbeddingService

        Returns:
            VectorDatabaseResult with operation details

        Raises:
            ValueError: If embedded_chunks is empty
            Exception: If database operation fails
        """
        if not embedded_chunks:
            raise ValueError("Cannot add empty list of embeddings")

        start_time = datetime.now()

        self.logger.info(f"Adding {len(embedded_chunks)} embeddings to database")

        try:
            # Prepare points for Qdrant
            points = []
            for embedded_chunk in embedded_chunks:
                chunk = embedded_chunk.chunk
                metadata = chunk.metadata

                # Generate unique ID
                point_id = str(uuid.uuid4())

                # Prepare payload (metadata)
                payload = {
                    "chunk_content": chunk.content,
                    "source_file": metadata.source_file,
                    "document_title": metadata.document_title,
                    "document_author": metadata.document_author,
                    "chunk_index": metadata.chunk_index,
                    "total_chunks": metadata.total_chunks,
                    "start_char": metadata.start_char,
                    "end_char": metadata.end_char,
                    "stored_at": datetime.now().isoformat(),
                }

                # Create point
                point = PointStruct(
                    id=hash(point_id) % (10**10),  # Convert to integer ID
                    vector=embedded_chunk.embedding,
                    payload=payload,
                )
                points.append(point)

            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.info(
                f"Successfully added {len(embedded_chunks)} embeddings " f"in {duration:.2f}s"
            )

            return VectorDatabaseResult(
                operation="add",
                success=True,
                items_affected=len(embedded_chunks),
                duration_seconds=duration,
            )

        except Exception as e:
            self.logger.error(f"Failed to add embeddings: {e}")
            return VectorDatabaseResult(
                operation="add",
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> VectorDatabaseResult:
        """
        Search for similar embeddings in the database.

        Args:
            query_embedding: Vector to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            VectorDatabaseResult with search results

        Raises:
            ValueError: If query_embedding dimension doesn't match
            Exception: If search fails
        """
        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} "
                f"doesn't match vector size {self.vector_size}"
            )

        start_time = datetime.now()

        self.logger.info(
            f"Searching for similar embeddings (limit={limit}, threshold={score_threshold})"
        )

        try:
            # Use query_points for Qdrant 1.17.1
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
            )

            # Convert results
            results = []
            for hit in search_results.points:
                score = hit.score if hasattr(hit, "score") else 0.0

                # Only include results above threshold
                if score >= score_threshold:
                    result = SearchResult(
                        point_id=str(hit.id),
                        chunk_content=hit.payload.get("chunk_content", ""),
                        score=score,
                        source_file=hit.payload.get("source_file", ""),
                        document_title=hit.payload.get("document_title"),
                        chunk_index=hit.payload.get("chunk_index", 0),
                    )
                    results.append(result)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.info(f"Found {len(results)} similar embeddings in {duration:.2f}s")

            return VectorDatabaseResult(
                operation="search",
                success=True,
                items_affected=len(results),
                results=results,
                duration_seconds=duration,
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return VectorDatabaseResult(
                operation="search",
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def delete_document(self, source_file: str) -> VectorDatabaseResult:
        """
        Delete all embeddings for a specific document.

        Args:
            source_file: Source file path to delete

        Returns:
            VectorDatabaseResult with deletion details

        Raises:
            Exception: If deletion fails
        """
        start_time = datetime.now()

        self.logger.info(f"Deleting embeddings for document: {source_file}")

        try:
            # Qdrant 1.17.1: scroll through points and delete those matching
            points_to_delete = []
            offset = 0
            while offset is not None:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                )

                for point in points:
                    if point.payload.get("source_file") == source_file:
                        points_to_delete.append(point.id)

            # Delete the identified points
            if points_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_to_delete,
                )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.info(f"Deleted {len(points_to_delete)} embeddings for {source_file}")

            return VectorDatabaseResult(
                operation="delete",
                success=True,
                items_affected=len(points_to_delete),
                duration_seconds=duration,
            )

        except Exception as e:
            self.logger.error(f"Failed to delete document {source_file}: {e}")
            return VectorDatabaseResult(
                operation="delete",
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "point_count": collection_info.points_count,
                "vector_size": self.vector_size,
                "persistence": self.persistence_path is not None,
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }

    def clear_all(self) -> VectorDatabaseResult:
        """
        Clear all embeddings from the database.

        WARNING: This is irreversible!

        Returns:
            VectorDatabaseResult indicating success
        """
        start_time = datetime.now()

        self.logger.warning("Clearing all embeddings from database")

        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return VectorDatabaseResult(
                operation="clear",
                success=True,
                duration_seconds=duration,
            )

        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")
            return VectorDatabaseResult(
                operation="clear",
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
