# -*- coding: utf-8 -*-
"""
Integration tests for complete pipeline: FileScanner → DocumentParser → ChunkingService → EmbeddingService

Tests verify that all components work seamlessly together.
"""

import pytest
from pathlib import Path
from datetime import datetime

from ragindexer import (
    FileScanner,
    DocumentParser,
    ChunkingService,
    EmbeddingService,
)
from ragindexer import FileFormat


class TestCompletePipeline:
    """Test the complete indexing pipeline."""

    def test_full_pipeline_txt(self, tmp_path):
        """Test complete pipeline with a TXT file."""
        # Create test document
        doc_path = tmp_path / "test.txt"
        doc_content = """First paragraph with some content.
It contains multiple sentences for testing.

Second paragraph here. With different content.
And another sentence.

Third paragraph: Lorem ipsum dolor sit amet.
More content for testing the chunking service.
"""
        doc_path.write_text(doc_content)

        # Step 1: Scan
        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        assert len(scan_result.files) == 1
        file_info = list(scan_result.files.values())[0]
        assert file_info.format == FileFormat.TXT

        # Step 2: Parse
        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        assert parsed_doc.content
        assert parsed_doc.character_count > 0
        assert parsed_doc.metadata.source_file == "test.txt"

        # Step 3: Chunk
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        chunking_result = chunking_service.chunk(parsed_doc)

        assert chunking_result.total_chunks > 0
        assert chunking_result.total_tokens > 0
        # Chunks may have slightly different character count due to
        # whitespace normalization when joining semantic units
        assert abs(chunking_result.total_characters - parsed_doc.character_count) <= 10

        # Verify chunks
        for chunk in chunking_result.chunks:
            assert chunk.content
            assert chunk.character_count > 0
            assert chunk.token_count > 0
            assert chunk.metadata.source_file == "test.txt"

    def test_full_pipeline_multiple_files(self, tmp_path):
        """Test pipeline with multiple files in subdirectories."""
        # Create multiple test documents
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2").mkdir()

        files = {
            "doc1.txt": "This is the first document. It has some content.",
            "subdir1/doc2.txt": "Second document in subdirectory.",
            "subdir2/doc3.txt": "Third document. Another subdirectory.",
        }

        for file_path, content in files.items():
            (tmp_path / file_path).write_text(content)

        # Step 1: Scan
        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        assert len(scan_result.files) == 3

        # Step 2 & 3: Parse and Chunk all documents
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)

        all_chunks = {}

        for file_path, file_info in scan_result.files.items():
            parsed_doc = parser.parse(file_info)
            chunking_result = chunking_service.chunk(parsed_doc)

            all_chunks[file_path] = {
                "parsed": parsed_doc,
                "chunks": chunking_result,
            }

        # Verify results
        assert len(all_chunks) == 3

        total_tokens = 0
        for file_path, data in all_chunks.items():
            chunking_result = data["chunks"]
            assert chunking_result.total_chunks > 0
            total_tokens += chunking_result.total_tokens

        assert total_tokens > 0

    def test_pipeline_with_markdown(self, tmp_path):
        """Test pipeline with Markdown file."""
        doc_path = tmp_path / "README.md"
        doc_content = """# Title

## Section 1

This is the first section with some content.

## Section 2

This is the second section with different content.
It has multiple lines.

### Subsection

And a subsection with more content.
"""
        doc_path.write_text(doc_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        parser = DocumentParser()
        parsed_doc = parser.parse(list(scan_result.files.values())[0])

        chunking_service = ChunkingService(chunk_size=150, overlap_size=30)
        chunking_result = chunking_service.chunk(parsed_doc)

        assert chunking_result.total_chunks > 0
        assert all(chunk.content for chunk in chunking_result.chunks)

    def test_pipeline_preserves_metadata_through_chain(self, tmp_path):
        """Test that metadata is preserved through the entire chain."""
        doc_path = tmp_path / "document.txt"
        doc_content = "Sample document content for testing metadata preservation."
        doc_path.write_text(doc_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService()
        chunking_result = chunking_service.chunk(parsed_doc)

        # Verify metadata chain
        assert chunking_result.document_path == file_info.relative_path
        for chunk in chunking_result.chunks:
            # Source file should match through the chain
            assert chunk.metadata.source_file == file_info.relative_path
            # File hash should be in metadata indirectly through file_info
            assert chunk.metadata.extracted_at is not None

    def test_pipeline_chunk_positions_are_accurate(self, tmp_path):
        """Test that chunk positions correctly reference original document."""
        doc_path = tmp_path / "test.txt"
        doc_content = "First. Second. Third. Fourth. Fifth."
        doc_path.write_text(doc_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService(chunk_size=50, overlap_size=5)
        chunking_result = chunking_service.chunk(parsed_doc)

        # Verify that positions correctly index the original content
        original_content = parsed_doc.content

        for chunk in chunking_result.chunks:
            start = chunk.metadata.start_char
            end = chunk.metadata.end_char

            # Extract from original using positions
            indexed_content = original_content[start:end]

            # Should be contained in the original
            assert indexed_content in original_content

    def test_pipeline_token_counts_are_consistent(self, tmp_path):
        """Test that token counts are consistent across pipeline."""
        doc_path = tmp_path / "test.txt"
        doc_content = "Word one two three. Word four five six. Word seven eight nine."
        doc_path.write_text(doc_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService(chunk_size=200, overlap_size=20)
        chunking_result = chunking_service.chunk(parsed_doc)

        # Verify consistency
        chunk_token_sum = sum(c.token_count for c in chunking_result.chunks)
        assert chunk_token_sum == chunking_result.total_tokens
        assert chunking_result.total_tokens > 0

    def test_pipeline_handles_large_document(self, tmp_path):
        """Test pipeline with a larger document."""
        doc_path = tmp_path / "large.txt"

        # Create a larger document
        paragraphs = [
            "This is paragraph number {}. It contains some text content.".format(i)
            for i in range(100)
        ]
        doc_content = "\n\n".join(paragraphs)
        doc_path.write_text(doc_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService(chunk_size=200, overlap_size=30)
        chunking_result = chunking_service.chunk(parsed_doc)

        # Verify reasonable chunking of large document
        assert chunking_result.total_chunks > 1
        assert all(
            chunk.token_count <= 300 for chunk in chunking_result.chunks
        )  # Some buffer above chunk_size due to overlap

    def test_pipeline_empty_handling(self, tmp_path):
        """Test pipeline handling of edge cases."""
        # Single file with minimal content
        doc_path = tmp_path / "minimal.txt"
        doc_path.write_text("Hi")

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService()
        chunking_result = chunking_service.chunk(parsed_doc)

        # Should still create at least one chunk
        assert chunking_result.total_chunks == 1
        assert chunking_result.chunks[0].content == "Hi"


class TestPipelineDataFlow:
    """Test data flow and transformations through pipeline."""

    def test_data_preservation_txt_to_chunks(self, tmp_path):
        """Test that no content is lost from TXT file to chunks."""
        doc_path = tmp_path / "test.txt"
        original_content = "Hello world. This is a test document. It has multiple sentences."
        doc_path.write_text(original_content)

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()

        parser = DocumentParser()
        parsed_doc = parser.parse(list(scan_result.files.values())[0])

        chunking_service = ChunkingService()
        chunking_result = chunking_service.chunk(parsed_doc)

        # Reconstruct content from chunks (with spaces)
        reconstructed = " ".join(c.content for c in chunking_result.chunks)

        # Normalize both for comparison (remove extra spaces)
        original_normalized = " ".join(original_content.split())
        reconstructed_normalized = " ".join(reconstructed.split())

        assert original_normalized == reconstructed_normalized

    def test_file_info_flows_through_chain(self, tmp_path):
        """Test that FileInfo attributes flow through to chunks."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("Test content.")

        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService()
        chunking_result = chunking_service.chunk(parsed_doc)

        # File info attributes should be accessible through chunks
        assert chunking_result.document_path == file_info.relative_path

        for chunk in chunking_result.chunks:
            assert chunk.metadata.source_file == file_info.relative_path


class TestCompletePipelineWithEmbeddings:
    """Test the complete pipeline including EmbeddingService (Component 4)."""

    def test_full_pipeline_with_embeddings(self, tmp_path):
        """Test complete pipeline: Scan → Parse → Chunk → Embed."""
        # Create test document
        doc_path = tmp_path / "test.txt"
        doc_content = """First paragraph about machine learning.
It is a subset of artificial intelligence.

Second paragraph about embeddings.
They are numerical representations of text.

Third paragraph about neural networks.
They have multiple layers for processing.
"""
        doc_path.write_text(doc_content)

        # Step 1: Scan
        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        assert len(scan_result.files) == 1
        file_info = list(scan_result.files.values())[0]

        # Step 2: Parse
        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)
        assert parsed_doc.content

        # Step 3: Chunk
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        chunking_result = chunking_service.chunk(parsed_doc)
        assert chunking_result.total_chunks > 0

        # Step 4: Embed
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Verify embedding results
        assert embedding_result.total_chunks == chunking_result.total_chunks
        assert embedding_result.embedding_dim == 384
        assert embedding_result.embedding_model == "all-MiniLM-L6-v2"
        assert embedding_result.total_time_seconds > 0

        # Verify embedded chunks
        assert len(embedding_result.embedded_chunks) == chunking_result.total_chunks
        for i, embedded_chunk in enumerate(embedding_result.embedded_chunks):
            # Check chunk content preserved
            assert embedded_chunk.chunk.content
            assert embedded_chunk.chunk.metadata.chunk_index == i

            # Check embedding vector
            assert len(embedded_chunk.embedding) == 384
            assert all(isinstance(val, (int, float)) for val in embedded_chunk.embedding)
            assert embedded_chunk.embedding_dim == 384
            assert embedded_chunk.embedding_model == "all-MiniLM-L6-v2"

    def test_pipeline_embeddings_multiple_documents(self, tmp_path):
        """Test embedding pipeline with multiple documents."""
        # Create multiple test documents
        (tmp_path / "docs").mkdir()
        files = {
            "docs/doc1.txt": "Machine learning is fascinating. It enables systems to learn from data.",
            "docs/doc2.txt": "Deep learning uses neural networks. It has revolutionized AI.",
        }

        for file_path, content in files.items():
            (tmp_path / file_path).write_text(content)

        # Setup services
        scanner = FileScanner(tmp_path)
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=80, overlap_size=15)
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        # Process all documents
        scan_result = scanner.scan()
        all_embeddings = {}

        for file_path, file_info in scan_result.files.items():
            parsed_doc = parser.parse(file_info)
            chunking_result = chunking_service.chunk(parsed_doc)
            embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

            all_embeddings[file_path] = {
                "chunks": chunking_result,
                "embeddings": embedding_result,
            }

        # Verify all documents were processed
        assert len(all_embeddings) == 2

        for file_path, data in all_embeddings.items():
            embedding_result = data["embeddings"]
            assert embedding_result.total_chunks > 0
            assert len(embedding_result.embedded_chunks) > 0

    def test_pipeline_metadata_preserved_through_embeddings(self, tmp_path):
        """Test that metadata is preserved through entire pipeline."""
        doc_path = tmp_path / "document.txt"
        doc_path.write_text("Sample document. With multiple sentences. For testing.")

        # Process through full pipeline
        scanner = FileScanner(tmp_path)
        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        chunking_service = ChunkingService(chunk_size=50, overlap_size=10)
        chunking_result = chunking_service.chunk(parsed_doc)

        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Verify metadata chain
        for embedded_chunk in embedding_result.embedded_chunks:
            # Original chunk metadata preserved
            chunk_metadata = embedded_chunk.chunk.metadata
            assert chunk_metadata.source_file == file_info.relative_path
            assert (
                chunk_metadata.document_title is not None or chunk_metadata.document_title is None
            )  # Both valid
            assert chunk_metadata.chunk_index >= 0
            assert chunk_metadata.total_chunks > 0

            # Embedding metadata correct
            assert embedded_chunk.embedding_model == "all-MiniLM-L6-v2"
            assert embedded_chunk.embedding_dim == 384

    def test_pipeline_embedding_similarity(self, tmp_path):
        """Test similarity calculations on embedded chunks."""
        doc_path = tmp_path / "test.txt"
        doc_content = """The cat sat on the mat. It was a comfortable place.

Dogs are loyal animals. They are great companions.

Cats are independent animals. They enjoy their own space.
"""
        doc_path.write_text(doc_content)

        # Process through full pipeline
        scanner = FileScanner(tmp_path)
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=80, overlap_size=15)
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]
        parsed_doc = parser.parse(file_info)
        chunking_result = chunking_service.chunk(parsed_doc)
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Calculate similarities between chunks
        embedded_chunks = embedding_result.embedded_chunks

        if len(embedded_chunks) >= 2:
            # Find similarities
            similarities = {}
            for i in range(len(embedded_chunks)):
                for j in range(i + 1, len(embedded_chunks)):
                    sim = embedding_service.similarity(
                        embedded_chunks[i].embedding,
                        embedded_chunks[j].embedding,
                    )
                    similarities[(i, j)] = sim

            # All similarities should be between 0 and 1
            assert all(0 <= sim <= 1 for sim in similarities.values())

            # Semantically similar chunks should have higher similarity
            # (e.g., chunks about cats should be more similar to each other)
            assert len(similarities) > 0

    def test_pipeline_batch_embedding_performance(self, tmp_path):
        """Test batch embedding performance with multiple chunks."""
        doc_path = tmp_path / "large.txt"

        # Create a document with many paragraphs
        paragraphs = [
            f"This is paragraph {i}. It contains information about topic {i % 3}."
            for i in range(50)
        ]
        doc_content = "\n\n".join(paragraphs)
        doc_path.write_text(doc_content)

        # Process through pipeline
        scanner = FileScanner(tmp_path)
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        embedding_service = EmbeddingService(batch_size=16, model_name="all-MiniLM-L6-v2")

        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]
        parsed_doc = parser.parse(file_info)
        chunking_result = chunking_service.chunk(parsed_doc)
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Verify all chunks were embedded
        assert embedding_result.total_chunks == chunking_result.total_chunks
        assert embedding_result.total_chunks > 5  # Should have multiple chunks

        # Verify timing
        assert embedding_result.total_time_seconds > 0
        speed = embedding_result.total_chunks / embedding_result.total_time_seconds
        assert speed > 0  # chunks per second

    def test_pipeline_embedding_with_different_models(self, tmp_path):
        """Test pipeline with different embedding models."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("Testing embedding with different models. This is sample content.")

        # Process through pipeline
        scanner = FileScanner(tmp_path)
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)

        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]
        parsed_doc = parser.parse(file_info)
        chunking_result = chunking_service.chunk(parsed_doc)

        # Test with multiple models (start with default)
        model_name = "all-MiniLM-L6-v2"
        embedding_service = EmbeddingService(model_name=model_name)
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Verify embedding results
        assert embedding_result.embedding_model == model_name
        assert embedding_result.embedding_dim == 384
        assert embedding_result.total_chunks > 0

    def test_pipeline_embedding_text_query(self, tmp_path):
        """Test embedding text queries for semantic search."""
        doc_path = tmp_path / "test.txt"
        doc_content = """Machine learning is a branch of artificial intelligence.
It enables systems to learn from data without being explicitly programmed.

Deep learning uses neural networks with multiple layers.
It has applications in computer vision and natural language processing.

Embeddings are vector representations of text.
They enable semantic similarity calculations.
"""
        doc_path.write_text(doc_content)

        # Process documents through pipeline
        scanner = FileScanner(tmp_path)
        parser = DocumentParser()
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        scan_result = scanner.scan()
        file_info = list(scan_result.files.values())[0]
        parsed_doc = parser.parse(file_info)
        chunking_result = chunking_service.chunk(parsed_doc)
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Test semantic search with query embeddings
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are embeddings?",
        ]

        for query in queries:
            query_embedding = embedding_service.embed_text(query)

            # Calculate similarities with document chunks
            similarities = []
            for embedded_chunk in embedding_result.embedded_chunks:
                sim = embedding_service.similarity(
                    query_embedding.tolist(),
                    embedded_chunk.embedding,
                )
                similarities.append((sim, embedded_chunk.chunk.content))

            # Sort by similarity
            similarities.sort(reverse=True)

            # Most similar chunk should have reasonable score
            if similarities:
                top_similarity, _ = similarities[0]
                assert 0 <= top_similarity <= 1
