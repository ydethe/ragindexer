# -*- coding: utf-8 -*-
"""
Embedding Service Component

Generates vector embeddings for document chunks using fastembed (ONNX-based).
Supports batch processing for performance optimization.
"""

import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

from ragindexer.ChunkingService import TextChunk

logger = logging.getLogger(__name__)


class EmbeddedChunk(BaseModel):
    """
    A text chunk with its generated embedding vector.

    Attributes:
        chunk: Original TextChunk
        embedding: Vector representation (list of floats)
        embedding_dim: Dimension of the embedding
        embedding_model: Model used to generate embedding
    """

    chunk: TextChunk
    embedding: List[float]
    embedding_dim: int
    embedding_model: str


class EmbeddingResult(BaseModel):
    """
    Result of embedding multiple chunks.

    Attributes:
        document_path: Source document path
        embedded_chunks: List of EmbeddedChunk objects
        total_chunks: Total number of chunks embedded
        embedding_model: Model used for embeddings
        embedding_dim: Dimension of embeddings
        total_time_seconds: Time taken for embedding (seconds)
        embedded_at: Timestamp of embedding
    """

    document_path: str
    embedded_chunks: List[EmbeddedChunk]
    total_chunks: int
    embedding_model: str
    embedding_dim: int
    total_time_seconds: float = 0.0
    embedded_at: datetime = Field(default_factory=datetime.now)


class EmbeddingService:
    """
    Service for generating vector embeddings for text chunks.

    Uses fastembed (ONNX-based) for fast, CPU-optimized embedding generation.
    Supports batch processing for performance.
    """

    # Cache for loaded models (class-level to avoid reloading)
    _model_cache = {}

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the EmbeddingService.

        Args:
            model_name: fastembed model ID (default: BAAI/bge-small-en-v1.5)
            batch_size: Batch size for embedding (default: 32)
            logger_instance: Logger to use (defaults to module logger)

        Note:
            Uses cached models to avoid reloading same model multiple times.
            fastembed downloads models automatically on first use.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger = logger_instance or logger

        # Load or get from cache
        if model_name not in self._model_cache:
            self.logger.info(f"Loading embedding model: {model_name}")
            try:
                model = TextEmbedding(model_name=model_name)
                # Determine embedding dimension by running a test embedding
                test_embedding = list(model.embed(["test"]))[0]
                dim = len(test_embedding)
                self._model_cache[model_name] = model
                self._model_cache[f"{model_name}_dim"] = dim
                self.logger.info(f"Model loaded successfully. Dimensions: {dim}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                raise

        self.model = self._model_cache[model_name]
        self.embedding_dim = self._model_cache[f"{model_name}_dim"]

    def embed_chunks(self, chunks: List[TextChunk]) -> EmbeddingResult:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of TextChunk objects to embed

        Returns:
            EmbeddingResult with embedded chunks and statistics

        Raises:
            ValueError: If chunks list is empty
            Exception: If embedding generation fails
        """
        if not chunks:
            raise ValueError("Cannot embed empty list of chunks")

        self.logger.info(f"Embedding {len(chunks)} chunks using model {self.model_name}")

        try:
            start_time = datetime.now()

            # Extract texts from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings with fastembed (returns a generator of numpy arrays)
            embeddings_list = [
                emb.tolist() for emb in self.model.embed(texts, batch_size=self.batch_size)
            ]

            # Create EmbeddedChunk objects
            embedded_chunks = [
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    embedding_dim=self.embedding_dim,
                    embedding_model=self.model_name,
                )
                for chunk, embedding in zip(chunks, embeddings_list)
            ]

            end_time = datetime.now()
            total_seconds = (end_time - start_time).total_seconds()

            # Get document path from first chunk
            document_path = chunks[0].metadata.source_file if chunks else "unknown"

            result = EmbeddingResult(
                document_path=document_path,
                embedded_chunks=embedded_chunks,
                total_chunks=len(embedded_chunks),
                embedding_model=self.model_name,
                embedding_dim=self.embedding_dim,
                total_time_seconds=total_seconds,
            )

            self.logger.info(
                f"Successfully embedded {result.total_chunks} chunks "
                f"in {total_seconds:.2f}s ({len(texts) / total_seconds:.1f} chunks/sec)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to embed chunks: {e}")
            raise

    def embed_single_chunk(self, chunk: TextChunk) -> EmbeddedChunk:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: TextChunk to embed

        Returns:
            EmbeddedChunk with embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        result = self.embed_chunks([chunk])
        return result.embedded_chunks[0]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for raw text.

        Useful for query embeddings in search operations.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding

        Raises:
            Exception: If embedding generation fails
        """
        try:
            embedding = list(self.model.embed([text]))[0]
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to embed text: {e}")
            raise

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions mismatch: {len(embedding1)} vs {len(embedding2)}"
            )

        # Convert to numpy arrays if needed
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def clear_cache(self):
        """Clear the model cache to free memory."""
        EmbeddingService._model_cache.clear()
        self.logger.info("Model cache cleared")
