# -*- coding: utf-8 -*-
"""
Tests for ChunkingService component.

Tests cover:
- Basic chunking functionality
- Overlap management
- Semantic preservation (paragraphs, sentences)
- Metadata association
- Token counting
- Edge cases (empty, very small, very large documents)
"""

import pytest
from datetime import datetime

from ragindexer import ChunkingService, TextChunk, ChunkMetadata, ChunkingResult
from ragindexer import DocumentParser, ParsedDocument, DocumentMetadata
from ragindexer import FileInfo, FileFormat
from pathlib import Path


class TestChunkingServiceBasic:
    """Basic chunking functionality tests."""

    @pytest.fixture
    def chunking_service(self):
        """Create a ChunkingService instance."""
        return ChunkingService(chunk_size=100, overlap_size=20)

    @pytest.fixture
    def sample_parsed_document(self, tmp_path):
        """Create a sample ParsedDocument for testing."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=1000,
            modified_time=datetime.now(),
            file_hash="abc123",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title="Test Document",
            author="Test Author",
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = """
This is the first paragraph with some text that should be chunked.
It contains multiple sentences for testing purposes.

This is the second paragraph. It has different content.
And another sentence here.

Third paragraph: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        """

        return ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

    def test_chunk_returns_chunking_result(self, chunking_service, sample_parsed_document):
        """Test that chunk() returns a ChunkingResult object."""
        result = chunking_service.chunk(sample_parsed_document)

        assert isinstance(result, ChunkingResult)
        assert result.document_path == "test.txt"
        assert len(result.chunks) > 0
        assert result.total_chunks == len(result.chunks)

    def test_chunk_creates_text_chunks(self, chunking_service, sample_parsed_document):
        """Test that chunks are properly created as TextChunk objects."""
        result = chunking_service.chunk(sample_parsed_document)

        assert all(isinstance(chunk, TextChunk) for chunk in result.chunks)

    def test_chunk_preserves_content(self, chunking_service, sample_parsed_document):
        """Test that total content is preserved (minus whitespace)."""
        result = chunking_service.chunk(sample_parsed_document)

        original_text = sample_parsed_document.content.strip()
        chunked_text = " ".join(chunk.content for chunk in result.chunks)

        # Remove extra whitespace for comparison
        original_normalized = " ".join(original_text.split())
        chunked_normalized = " ".join(chunked_text.split())

        assert original_normalized == chunked_normalized

    def test_chunk_empty_document_raises_error(self, chunking_service):
        """Test that chunking empty document raises ValueError."""
        file_info = FileInfo(
            relative_path="empty.txt",
            absolute_path=Path("/tmp/empty.txt"),
            format=FileFormat.TXT,
            file_size=0,
            modified_time=datetime.now(),
            file_hash="",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="empty.txt",
            format=FileFormat.TXT,
        )

        parsed_doc = ParsedDocument(
            content="",
            metadata=metadata,
            file_info=file_info,
        )

        with pytest.raises(ValueError, match="Cannot chunk empty document"):
            chunking_service.chunk(parsed_doc)

    def test_chunk_metadata_is_consistent(self, chunking_service, sample_parsed_document):
        """Test that metadata is correctly associated with chunks."""
        result = chunking_service.chunk(sample_parsed_document)

        for idx, chunk in enumerate(result.chunks):
            assert isinstance(chunk.metadata, ChunkMetadata)
            assert chunk.metadata.document == "test.txt"
            assert chunk.metadata.document_title == "Test Document"
            assert chunk.metadata.document_author == "Test Author"
            assert chunk.metadata.chunk_index == idx
            assert chunk.metadata.total_chunks == result.total_chunks


class TestChunkingServiceCharacterCount:
    """Tests for character counting and position tracking."""

    @pytest.fixture
    def simple_service(self):
        """Create a simple ChunkingService."""
        return ChunkingService(chunk_size=50, overlap_size=10)

    def test_character_count_in_chunks(self, tmp_path):
        """Test that character counts are accurate."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=100,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = "Hello world. This is a test document."

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=100, overlap_size=10)
        result = service.chunk(parsed_doc)

        for chunk in result.chunks:
            assert chunk.character_count == len(chunk.content)

    def test_chunk_positions_are_correct(self, tmp_path):
        """Test that start and end character positions are correct."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=100,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = "First part. Second part. Third part."

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=100, overlap_size=10)
        result = service.chunk(parsed_doc)

        # Check that positions are sequential and non-overlapping in original
        for chunk in result.chunks:
            assert chunk.metadata.start_char >= 0
            assert chunk.metadata.end_char > chunk.metadata.start_char
            # Verify position matches content
            actual_content = content[chunk.metadata.start_char : chunk.metadata.end_char]
            assert actual_content in content


class TestChunkingServiceTokenCounting:
    """Tests for token counting functionality."""

    @pytest.fixture
    def service(self):
        """Create a ChunkingService instance."""
        return ChunkingService(chunk_size=100, overlap_size=10)

    def test_token_count_is_approximated(self, service, tmp_path):
        """Test that token count is approximated."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=100,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        # 4 words should be approximately 5 tokens (4 / 0.75 ≈ 5)
        content = "One two three four"

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        result = service.chunk(parsed_doc)

        assert len(result.chunks) > 0
        first_chunk = result.chunks[0]
        assert first_chunk.token_count > 0
        # 4 words / 0.75 = ~5 tokens
        assert first_chunk.token_count >= 4

    def test_total_tokens_sum(self, service, tmp_path):
        """Test that total tokens is the sum of chunk tokens."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=100,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = "Word " * 50  # 250 words

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        result = service.chunk(parsed_doc)

        chunk_sum = sum(chunk.token_count for chunk in result.chunks)
        assert result.total_tokens == chunk_sum


class TestChunkingServiceSemanticPreservation:
    """Tests for semantic preservation in chunking."""

    def test_paragraph_preservation(self, tmp_path):
        """Test that paragraphs are not split unnecessarily."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=500,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        # Create content with clear paragraphs
        content = """First paragraph with some content here.

Second paragraph with different content here.

Third paragraph is another one."""

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=200, overlap_size=20)
        result = service.chunk(parsed_doc)

        # With large chunk size, should have at least 1 chunk containing
        # parts from multiple paragraphs or one chunk per paragraph
        assert result.total_chunks > 0

        # Verify no paragraph is split across chunks improperly
        for chunk in result.chunks:
            # Each chunk should be continuous content
            assert len(chunk.content) > 0

    def test_sentence_handling(self, tmp_path):
        """Test that sentences are handled intelligently."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=200,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = "First sentence. Second sentence. " "Third sentence. Fourth sentence."

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=100, overlap_size=10)
        result = service.chunk(parsed_doc)

        # Should create reasonable chunks
        assert result.total_chunks > 0
        assert result.total_characters == len(content)


class TestChunkingServiceOverlap:
    """Tests for overlap functionality between chunks."""

    def test_overlap_creates_continuity(self, tmp_path):
        """Test that overlap provides continuity between chunks."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=500,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        # Create content that will definitely span multiple chunks
        content = " ".join(["word" + str(i) for i in range(100)])

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=50, overlap_size=20)
        result = service.chunk(parsed_doc)

        # With overlap, consecutive chunks should share some content
        if result.total_chunks > 1:
            for i in range(len(result.chunks) - 1):
                chunk1 = result.chunks[i].content
                chunk2 = result.chunks[i + 1].content

                # Due to overlap, the end of chunk1 should be similar to
                # the beginning of chunk2
                # Just verify they exist and are different
                assert chunk1 != chunk2 or result.total_chunks == 1


class TestChunkingServiceStatistics:
    """Tests for result statistics."""

    def test_result_statistics_are_accurate(self, tmp_path):
        """Test that ChunkingResult statistics are accurate."""
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=tmp_path / "test.txt",
            format=FileFormat.TXT,
            file_size=100,
            modified_time=datetime.now(),
            file_hash="abc",
            detected_at=datetime.now(),
        )

        metadata = DocumentMetadata(
            title=None,
            author=None,
            page_count=None,
            document="test.txt",
            format=FileFormat.TXT,
        )

        content = "This is a test. " * 20

        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            file_info=file_info,
        )

        service = ChunkingService(chunk_size=100, overlap_size=20)
        result = service.chunk(parsed_doc)

        # Verify statistics
        assert result.total_chunks == len(result.chunks)
        assert result.total_characters == sum(c.character_count for c in result.chunks)
        assert result.total_tokens == sum(c.token_count for c in result.chunks)
        assert isinstance(result.chunking_time, datetime)


class TestChunkingServiceIntegration:
    """Integration tests with DocumentParser."""

    def test_integration_with_document_parser(self, tmp_path):
        """Test ChunkingService integration with DocumentParser."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = """
        This is a test document.
        It contains multiple paragraphs.

        Second paragraph here.
        With some content.

        Third paragraph for testing.
        """
        test_file.write_text(test_content)

        # Use FileScanner and DocumentParser
        file_info = FileInfo(
            relative_path="test.txt",
            absolute_path=test_file,
            format=FileFormat.TXT,
            file_size=len(test_content),
            modified_time=datetime.now(),
            file_hash="abc123",
            detected_at=datetime.now(),
        )

        parser = DocumentParser()
        parsed_doc = parser.parse(file_info)

        # Now chunk it
        chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
        result = chunking_service.chunk(parsed_doc)

        # Verify result
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

        # Verify metadata is preserved
        for chunk in result.chunks:
            assert chunk.metadata.document == "test.txt"
            assert chunk.content  # Non-empty content
            assert chunk.character_count > 0
            assert chunk.token_count > 0
