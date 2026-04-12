# -*- coding: utf-8 -*-
"""
Tests for EmbeddingService component.

Tests cover:
- Basic embedding functionality
- Batch processing
- Model caching
- Similarity calculations
- Integration with ChunkingService
- Error handling
"""

import pytest
from datetime import datetime

from ragindexer import (
    EmbeddingService,
    EmbeddedChunk,
    EmbeddingResult,
)
from ragindexer import (
    ChunkingService,
    TextChunk,
    ChunkMetadata,
)
from ragindexer import (
    DocumentParser,
    FileScanner,
)


class TestEmbeddingServiceBasic:
    """Basic embedding functionality tests."""

    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance."""
        return EmbeddingService(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=32,
        )

    @pytest.fixture
    def sample_chunks(self):
        """Create sample TextChunk objects."""
        chunks = []

        texts = [
            "This is the first chunk of text.",
            "This is the second chunk of text.",
            "This is the third chunk of text.",
        ]

        for idx, text in enumerate(texts):
            metadata = ChunkMetadata(
                document="test.txt",
                document_title="Test Document",
                document_author="Test Author",
                chunk_index=idx,
                total_chunks=len(texts),
                start_char=idx * 50,
                end_char=(idx + 1) * 50,
            )

            chunk = TextChunk(
                content=text,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def test_embed_chunks_returns_result(self, embedding_service, sample_chunks):
        """Test that embed_chunks returns EmbeddingResult."""
        result = embedding_service.embed_chunks(sample_chunks)

        assert isinstance(result, EmbeddingResult)
        assert result.total_chunks == len(sample_chunks)
        assert result.embedding_model == "BAAI/bge-small-en-v1.5"
        assert result.embedding_dim == 384  # MiniLM dimension

    def test_embed_chunks_creates_embeddings(self, embedding_service, sample_chunks):
        """Test that embeddings are created."""
        result = embedding_service.embed_chunks(sample_chunks)

        assert len(result.embedded_chunks) == len(sample_chunks)

        for embedded_chunk in result.embedded_chunks:
            assert isinstance(embedded_chunk, EmbeddedChunk)
            assert len(embedded_chunk.embedding) == 384
            assert embedded_chunk.embedding_dim == 384

    def test_embedding_vectors_are_numeric(self, embedding_service, sample_chunks):
        """Test that embeddings contain valid numeric values."""
        result = embedding_service.embed_chunks(sample_chunks)

        for embedded_chunk in result.embedded_chunks:
            # Check all values are numbers
            assert all(isinstance(val, (int, float)) for val in embedded_chunk.embedding)

            # Check values are in reasonable range
            assert all(-10 < val < 10 for val in embedded_chunk.embedding)

    def test_embed_empty_chunks_raises_error(self, embedding_service):
        """Test that embedding empty chunk list raises error."""
        with pytest.raises(ValueError, match="Cannot embed empty"):
            embedding_service.embed_chunks([])

    def test_chunk_metadata_preserved(self, embedding_service, sample_chunks):
        """Test that chunk metadata is preserved in result."""
        result = embedding_service.embed_chunks(sample_chunks)

        for original_chunk, embedded_chunk in zip(sample_chunks, result.embedded_chunks):
            assert embedded_chunk.chunk == original_chunk
            assert embedded_chunk.chunk.metadata.document == "test.txt"

    def test_embedding_time_tracked(self, embedding_service, sample_chunks):
        """Test that embedding time is tracked."""
        result = embedding_service.embed_chunks(sample_chunks)

        assert result.total_time_seconds > 0
        assert isinstance(result.embedded_at, datetime)

    def test_embedding_deterministic(self, embedding_service, sample_chunks):
        """Test that same input produces consistent embeddings."""
        result1 = embedding_service.embed_chunks(sample_chunks)
        result2 = embedding_service.embed_chunks(sample_chunks)

        # Compare first embeddings
        emb1 = result1.embedded_chunks[0].embedding
        emb2 = result2.embedded_chunks[0].embedding

        # Should be very close (may have minor floating point differences)
        for v1, v2 in zip(emb1, emb2):
            assert abs(v1 - v2) < 1e-5


class TestEmbeddingServiceSingleChunk:
    """Tests for single chunk embedding."""

    def test_embed_single_chunk(self):
        """Test embedding a single chunk."""
        service = EmbeddingService()

        metadata = ChunkMetadata(
            document="test.txt",
            document_title=None,
            document_author=None,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=30,
        )

        chunk = TextChunk(
            content="This is a test chunk.",
            metadata=metadata,
        )

        embedded = service.embed_single_chunk(chunk)

        assert isinstance(embedded, EmbeddedChunk)
        assert len(embedded.embedding) == 384
        assert embedded.chunk == chunk


class TestEmbeddingServiceTextEmbedding:
    """Tests for raw text embedding."""

    def test_embed_text(self):
        """Test embedding raw text."""
        service = EmbeddingService()

        text = "This is a test text for embedding."
        embedding = service.embed_text(text)

        assert embedding is not None
        assert len(embedding) == 384

    def test_embed_text_returns_numpy(self):
        """Test that embed_text returns numpy array."""
        import numpy as np

        service = EmbeddingService()
        embedding = service.embed_text("Test text")

        assert isinstance(embedding, np.ndarray)


class TestEmbeddingServiceSimilarity:
    """Tests for similarity calculations."""

    @pytest.fixture
    def service(self):
        """Create EmbeddingService."""
        return EmbeddingService()

    def test_similarity_identical_embeddings(self, service):
        """Test similarity of identical embeddings."""
        embedding = [1.0, 2.0, 3.0]
        similarity = service.similarity(embedding, embedding)

        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_similarity_perpendicular_embeddings(self, service):
        """Test similarity of perpendicular embeddings."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        similarity = service.similarity(embedding1, embedding2)

        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_similarity_dimension_mismatch_raises_error(self, service):
        """Test that dimension mismatch raises error."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0]

        with pytest.raises(ValueError, match="dimension"):
            service.similarity(embedding1, embedding2)

    def test_similarity_semantic(self, service):
        """Test similarity with semantic embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A feline rested on the rug."
        text3 = "The weather is sunny today."

        emb1 = service.embed_text(text1).tolist()
        emb2 = service.embed_text(text2).tolist()
        emb3 = service.embed_text(text3).tolist()

        # Semantically similar texts should have higher similarity
        sim_semantic = service.similarity(emb1, emb2)
        sim_different = service.similarity(emb1, emb3)

        assert sim_semantic > sim_different


class TestEmbeddingServiceCaching:
    """Tests for model caching."""

    def test_model_cache_same_model(self):
        """Test that same model is cached."""
        service1 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
        service2 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

        # Should be same instance due to cache
        assert service1.model is service2.model

    def test_clear_cache(self):
        """Test cache clearing."""
        service = EmbeddingService()
        # initial_cache_size = len(EmbeddingService._model_cache)

        service.clear_cache()

        assert len(EmbeddingService._model_cache) == 0


class TestEmbeddingServiceIntegration:
    """Integration tests with ChunkingService."""

    def test_integration_with_chunking_service(self, tmp_path):
        """Test integration with ChunkingService."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = """First paragraph with content.
It has multiple sentences.

Second paragraph here.
With more content."""
        test_file.write_text(content)

        # Step 1: Scan
        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        # Step 2: Parse
        parser = DocumentParser()
        file_info = list(scan_result.files.values())[0]
        parsed_doc = parser.parse(file_info)

        # Step 3: Chunk
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        chunking_result = chunking_service.chunk(parsed_doc)

        # Step 4: Embed
        embedding_service = EmbeddingService()
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Verify
        assert embedding_result.total_chunks > 0
        assert len(embedding_result.embedded_chunks) == chunking_result.total_chunks

        for i, embedded_chunk in enumerate(embedding_result.embedded_chunks):
            assert len(embedded_chunk.embedding) == 384
            assert embedded_chunk.chunk.metadata.chunk_index == i

    def test_full_pipeline_embedding(self, tmp_path):
        """Test complete pipeline including embedding."""
        # Create multiple test files
        (tmp_path / "doc1.txt").write_text("First document content. " * 20)
        (tmp_path / "doc2.txt").write_text("Second document content. " * 20)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=200, overlap_size=30)
        embedding_service = EmbeddingService()

        total_embedded = 0

        for file_path, file_info in scan_result.files.items():
            parsed_doc = parser.parse(file_info)
            chunking_result = chunking_service.chunk(parsed_doc)
            embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

            total_embedded += embedding_result.total_chunks

        assert total_embedded > 0


class TestEmbeddingServiceBatching:
    """Tests for batch processing."""

    def test_batch_processing(self):
        """Test batch processing of chunks."""
        service = EmbeddingService(batch_size=2)

        chunks = []
        for i in range(5):
            metadata = ChunkMetadata(
                document="test.txt",
                chunk_index=i,
                total_chunks=5,
                start_char=i * 20,
                end_char=(i + 1) * 20,
            )
            chunk = TextChunk(
                content=f"Chunk number {i} with content.",
                metadata=metadata,
            )
            chunks.append(chunk)

        result = service.embed_chunks(chunks)

        assert result.total_chunks == 5
        assert len(result.embedded_chunks) == 5


class TestEmbeddingServiceDocumentPath:
    """Tests for document path tracking."""

    def test_document_path_in_result(self):
        """Test that document path is included in result."""
        service = EmbeddingService()

        metadata = ChunkMetadata(
            document="documents/test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=30,
        )

        chunk = TextChunk(
            content="Test content.",
            metadata=metadata,
        )

        result = service.embed_chunks([chunk])

        assert result.document_path == "documents/test.txt"


class TestEmbeddingServiceErrorHandling:
    """Tests for error handling."""

    def test_invalid_text_handling(self):
        """Test handling of edge case texts."""
        service = EmbeddingService()

        # Empty string
        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=0,
        )

        chunk = TextChunk(
            content="",
            metadata=metadata,
        )

        # Should still work (will generate an embedding for empty string)
        result = service.embed_chunks([chunk])
        assert result.total_chunks == 1

    def test_very_long_text(self):
        """Test handling of very long text."""
        service = EmbeddingService()

        long_text = "This is a test. " * 1000  # Very long text

        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len(long_text),
        )

        chunk = TextChunk(
            content=long_text,
            metadata=metadata,
        )

        result = service.embed_chunks([chunk])

        assert result.total_chunks == 1
        assert len(result.embedded_chunks[0].embedding) == 384
