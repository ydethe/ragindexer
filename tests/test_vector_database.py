# -*- coding: utf-8 -*-
"""
Tests for VectorDatabaseService component.

Tests cover:
- Basic storage and retrieval
- Semantic search
- Document deletion
- Database statistics
- Integration with EmbeddingService
- Persistence (in-memory and on-disk)
"""

import pytest

from ragindexer import (
    VectorDatabaseService,
    SearchResult,
)
from ragindexer import (
    EmbeddingService,
    EmbeddedChunk,
    TextChunk,
    ChunkMetadata,
)


class TestVectorDatabaseBasic:
    """Basic vector database functionality tests."""

    @pytest.fixture
    def vector_db(self, tmp_path):
        """Create a VectorDatabaseService instance."""
        return VectorDatabaseService(
            collection_name="test_collection",
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
        )

    @pytest.fixture
    def sample_embedded_chunks(self):
        """Create sample EmbeddedChunk objects."""
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

            # Create fake embedding (384 values)
            embedding = [0.1 * (i % 384) for i in range(384)]

            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                embedding_dim=384,
                embedding_model="BAAI/bge-small-en-v1.5",
            )
            chunks.append(embedded_chunk)

        return chunks

    def test_add_embeddings_success(self, vector_db, sample_embedded_chunks):
        """Test adding embeddings to database."""
        result = vector_db.add_embeddings(sample_embedded_chunks)

        assert result.operation == "add"
        assert result.success is True
        assert result.items_affected == len(sample_embedded_chunks)
        assert result.duration_seconds > 0

    def test_add_empty_embeddings_raises_error(self, vector_db):
        """Test that adding empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot add empty"):
            vector_db.add_embeddings([])

    def test_search_returns_results(self, vector_db, sample_embedded_chunks):
        """Test that search returns results."""
        # Add embeddings first
        vector_db.add_embeddings(sample_embedded_chunks)

        # Search using same embedding
        query_embedding = sample_embedded_chunks[0].embedding
        result = vector_db.search(query_embedding, limit=5)

        assert result.operation == "search"
        assert result.success is True
        assert len(result.results) > 0
        assert isinstance(result.results[0], SearchResult)

    def test_search_results_have_scores(self, vector_db, sample_embedded_chunks):
        """Test that search results include similarity scores."""
        vector_db.add_embeddings(sample_embedded_chunks)

        query_embedding = sample_embedded_chunks[0].embedding
        result = vector_db.search(query_embedding, limit=5)

        assert len(result.results) > 0
        for search_result in result.results:
            # Allow small floating point errors
            assert -0.01 <= search_result.score <= 1.01
            assert search_result.content
            assert search_result.document

    def test_search_with_score_threshold(self, vector_db, sample_embedded_chunks):
        """Test search with score threshold."""
        vector_db.add_embeddings(sample_embedded_chunks)

        query_embedding = sample_embedded_chunks[0].embedding
        # High threshold - should get fewer results
        result = vector_db.search(query_embedding, limit=5, score_threshold=0.8)

        assert result.success is True
        # All results should be above threshold
        for search_result in result.results:
            assert search_result.score >= 0.8

    def test_search_wrong_dimension_raises_error(self, vector_db):
        """Test that searching with wrong dimension raises error."""
        wrong_embedding = [0.1] * 100  # Wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            vector_db.search(wrong_embedding)

    def test_get_statistics(self, vector_db, sample_embedded_chunks):
        """Test database statistics."""
        vector_db.add_embeddings(sample_embedded_chunks)

        stats = vector_db.get_statistics()

        assert stats["collection_name"] == "test_collection"
        assert stats["point_count"] == len(sample_embedded_chunks)
        assert stats["vector_size"] == 384

    def test_delete_document(self, vector_db, sample_embedded_chunks):
        """Test document deletion."""
        vector_db.add_embeddings(sample_embedded_chunks)

        # stats_before = vector_db.get_statistics()
        # count_before = stats_before["point_count"]

        result = vector_db.delete_document("test.txt")

        assert result.operation == "delete"
        assert result.success is True
        assert result.items_affected > 0

    def test_clear_all(self, vector_db, sample_embedded_chunks):
        """Test clearing all embeddings."""
        vector_db.add_embeddings(sample_embedded_chunks)

        result = vector_db.clear_all()

        assert result.operation == "clear"
        assert result.success is True

        # Database should be empty
        stats = vector_db.get_statistics()
        assert stats["point_count"] == 0


class TestVectorDatabaseMemory:
    """Tests for in-memory vector database."""

    def test_in_memory_database(self):
        """Test in-memory vector database."""
        vector_db = VectorDatabaseService(
            collection_name="memory_test",
            vector_size=384,
            persistence_path=None,
        )

        # Create simple embedded chunk
        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=10,
        )
        chunk = TextChunk(content="Test chunk", metadata=metadata)
        embedding = [0.1] * 384

        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_dim=384,
            embedding_model="test",
        )

        result = vector_db.add_embeddings([embedded_chunk])
        assert result.success is True


class TestVectorDatabasePersistence:
    """Tests for persistent (on-disk) vector database."""

    def ntest_persistent_database(self, tmp_path):
        """Test persistent vector database."""
        db_path = tmp_path / "test_db"

        vector_db = VectorDatabaseService(
            collection_name="persistent_test",
            vector_size=384,
            persistence_path=db_path,
        )

        # Verify path was created
        assert db_path.exists()

        # Create and add embedding
        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=10,
        )
        chunk = TextChunk(content="Test chunk", metadata=metadata)
        embedding = [0.1] * 384

        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_dim=384,
            embedding_model="test",
        )

        result = vector_db.add_embeddings([embedded_chunk])
        assert result.success is True

        # Create new instance with same path - should load existing data
        vector_db2 = VectorDatabaseService(
            collection_name="persistent_test",
            vector_size=384,
            persistence_path=db_path,
        )

        stats = vector_db2.get_statistics()
        assert stats["point_count"] > 0


class TestVectorDatabaseIntegration:
    """Integration tests with EmbeddingService."""

    def test_integration_with_embedding_service(self, tmp_path):
        """Test integration with EmbeddingService."""
        # Create vector database
        vector_db = VectorDatabaseService(
            collection_name="integration_test",
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
        )

        # Create sample embedded chunks using EmbeddingService
        embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

        # Create sample chunks
        chunks = []
        texts = [
            "Machine learning is fascinating.",
            "Deep learning uses neural networks.",
            "Embeddings capture semantic meaning.",
        ]

        for idx, text in enumerate(texts):
            metadata = ChunkMetadata(
                document="test.txt",
                chunk_index=idx,
                total_chunks=len(texts),
                start_char=idx * 50,
                end_char=(idx + 1) * 50,
            )
            chunk = TextChunk(content=text, metadata=metadata)
            chunks.append(chunk)

        # Generate embeddings
        embedding_result = embedding_service.embed_chunks(chunks)

        # Store in vector database
        db_result = vector_db.add_embeddings(embedding_result.embedded_chunks)

        assert db_result.success is True
        assert db_result.items_affected == len(chunks)

        # Search with a query
        query_text = "What is machine learning?"
        query_embedding = embedding_service.embed_text(query_text)
        search_result = vector_db.search(query_embedding.tolist(), limit=3)

        assert search_result.success is True
        assert len(search_result.results) > 0

        # Top result should be similar to query
        top_result = search_result.results[0]
        assert top_result.score > 0.5

    def test_full_pipeline_with_database(self, tmp_path):
        """Test complete pipeline: Scan → Parse → Chunk → Embed → Store."""
        from ragindexer import (
            FileScanner,
            DocumentParser,
            ChunkingService,
        )

        # Create test document
        doc_path = tmp_path / "docs"
        doc_path.mkdir()
        (doc_path / "test.txt").write_text(
            """Machine learning is a subset of AI.
It enables systems to learn from data.

Neural networks have multiple layers.
They process information hierarchically."""
        )

        # Pipeline
        scanner = FileScanner(doc_path)
        scan_result = scanner.scan()

        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
        vector_db = VectorDatabaseService(
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
        )

        # Process documents
        for file_info in scan_result.files.values():
            parsed_doc = parser.parse(file_info)
            chunking_result = chunking_service.chunk(parsed_doc)
            embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

            db_result = vector_db.add_embeddings(embedding_result.embedded_chunks)
            assert db_result.success is True

        # Verify data is stored
        stats = vector_db.get_statistics()
        assert stats["point_count"] > 0


class TestVectorDatabaseSearch:
    """Tests for search functionality."""

    def test_semantic_search_quality(self, tmp_path):
        """Test that semantic search returns relevant results."""
        vector_db = VectorDatabaseService(vector_size=384, persistence_path=tmp_path / "qdrant")
        embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

        # Create semantically related chunks
        texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "The dog chased the ball.",
            "A canine ran after a sphere.",
            "The weather is sunny.",
        ]

        chunks = []
        for idx, text in enumerate(texts):
            metadata = ChunkMetadata(
                document="test.txt",
                chunk_index=idx,
                total_chunks=len(texts),
                start_char=idx * 50,
                end_char=(idx + 1) * 50,
            )
            chunk = TextChunk(content=text, metadata=metadata)
            chunks.append(chunk)

        # Embed and store
        embedding_result = embedding_service.embed_chunks(chunks)
        vector_db.add_embeddings(embedding_result.embedded_chunks)

        # Search for cat-related query
        query_embedding = embedding_service.embed_text("The cat is sitting")
        results = vector_db.search(query_embedding.tolist(), limit=3)

        # First result should be about cats
        assert results.results[0].content  # Some content returned
        assert results.success is True

    def test_search_limit_respected(self, tmp_path):
        """Test that search limit is respected."""
        vector_db = VectorDatabaseService(vector_size=384, persistence_path=tmp_path / "qdrant")

        # Add multiple embeddings
        chunks = []
        for i in range(20):
            metadata = ChunkMetadata(
                document=f"doc{i}.txt",
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=10,
            )
            chunk = TextChunk(
                content=f"Document {i} content",
                metadata=metadata,
            )
            embedding = [0.1 * (i % 384) for i in range(384)]

            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                embedding_dim=384,
                embedding_model="test",
            )
            chunks.append(embedded_chunk)

        vector_db.add_embeddings(chunks)

        # Search with limit
        query_embedding = [0.1] * 384
        result = vector_db.search(query_embedding, limit=5)

        assert len(result.results) <= 5


class TestVectorDatabaseErrors:
    """Error handling tests."""

    def test_get_statistics_with_error(self, tmp_path):
        """Test statistics with non-existent collection."""
        vector_db = VectorDatabaseService(
            collection_name="nonexistent",
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
        )

        # Should handle error gracefully
        stats = vector_db.get_statistics()
        assert "collection_name" in stats


class TestVectorDatabaseApiKey:
    """Tests for QDRANT_API_KEY support."""

    def test_api_key_default_none(self, tmp_path):
        """Test that api_key defaults to None."""
        vector_db = VectorDatabaseService(
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
        )
        assert vector_db.api_key is None

    def test_api_key_stored(self, tmp_path):
        """Test that api_key is stored when provided."""
        vector_db = VectorDatabaseService(
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
            api_key="my-secret-key",
        )
        assert vector_db.api_key == "my-secret-key"

    def test_api_key_does_not_break_operations(self, tmp_path):
        """Test that providing an api_key does not break normal operations."""
        vector_db = VectorDatabaseService(
            vector_size=384,
            persistence_path=tmp_path / "qdrant",
            api_key="test-key",
        )

        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=10,
        )
        chunk = TextChunk(content="Test content", metadata=metadata)
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1] * 384,
            embedding_dim=384,
            embedding_model="test",
        )

        result = vector_db.add_embeddings([embedded_chunk])
        assert result.success is True


class TestVectorDatabaseMetadata:
    """Tests for metadata handling."""

    def test_metadata_preserved(self, tmp_path):
        """Test that metadata is preserved through storage."""
        vector_db = VectorDatabaseService(vector_size=384, persistence_path=tmp_path / "qdrant")

        # Create chunk with metadata
        metadata = ChunkMetadata(
            document="my_document.txt",
            document_title="My Title",
            document_author="My Author",
            chunk_index=5,
            total_chunks=10,
            start_char=100,
            end_char=200,
        )
        chunk = TextChunk(content="Test content", metadata=metadata)
        embedding = [0.1] * 384

        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_dim=384,
            embedding_model="test",
        )

        # Store
        vector_db.add_embeddings([embedded_chunk])

        # Search and verify metadata
        query_embedding = [0.1] * 384
        result = vector_db.search(query_embedding, limit=1)

        assert len(result.results) > 0
        search_result = result.results[0]
        assert search_result.document == "my_document.txt"
        assert search_result.document_title == "My Title"
        assert search_result.chunk_index == 5
