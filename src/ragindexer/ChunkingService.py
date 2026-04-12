# -*- coding: utf-8 -*-
"""
Chunking Service Component

Splits parsed documents into optimized chunks for embedding generation.
Handles intelligent chunking with overlap, preserving semantic boundaries.
"""

import re
import logging
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ragindexer.DocumentParser import ParsedDocument

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """
    Metadata associated with a text chunk.

    Attributes:
        document: Original file path
        document_title: Document title if available
        document_author: Document author if available
        chunk_index: Sequential index of chunk in document
        total_chunks: Total number of chunks in document
        start_char: Character position where chunk starts in original text
        end_char: Character position where chunk ends in original text
        extracted_at: When the chunk was created
    """

    document: str
    document_title: Optional[str] = None
    document_author: Optional[str] = None
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    extracted_at: datetime = Field(default_factory=datetime.now)


class TextChunk(BaseModel):
    """
    A chunk of text ready for embedding.

    Attributes:
        content: The actual text content
        metadata: Chunk-specific metadata
        character_count: Total characters in this chunk
        token_count: Approximate token count (for budgeting)
    """

    content: str
    metadata: ChunkMetadata
    character_count: int = Field(default=0)
    token_count: int = Field(default=0)

    def __init__(self, **data):
        super().__init__(**data)
        self.character_count = len(self.content)
        # Approximate token count: 1 token ≈ 0.75 words
        word_count = len(self.content.split())
        self.token_count = max(1, round(word_count / 0.75))


class ChunkingResult(BaseModel):
    """
    Result of chunking a document.

    Attributes:
        document_path: Source document path
        chunks: List of created chunks
        total_chunks: Total number of chunks
        total_characters: Total characters processed
        total_tokens: Total tokens across all chunks
        chunking_time: When chunking was performed
    """

    document_path: str
    chunks: List[TextChunk]
    total_chunks: int
    total_characters: int
    total_tokens: int
    chunking_time: datetime = Field(default_factory=datetime.now)


class ChunkingService:
    """
    Service for splitting parsed documents into semantically coherent chunks.

    Handles:
    - Intelligent chunking (preserving paragraphs and sentences)
    - Overlap management for semantic continuity
    - Metadata association for each chunk
    - Token counting for embedding budgeting
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 50,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ChunkingService.

        Args:
            chunk_size: Target size of chunks in tokens (default 512)
            overlap_size: Overlap size in tokens between consecutive chunks (default 50)
            logger_instance: Logger to use (defaults to module logger)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.logger = logger_instance or logger

    def chunk(self, parsed_document: ParsedDocument) -> ChunkingResult:
        """
        Split a parsed document into chunks.

        Strategy:
        1. Split by paragraphs
        2. If paragraph > chunk_size, split by sentences
        3. If sentence > chunk_size, split by words
        4. Apply overlap between chunks
        5. Associate metadata with each chunk

        Args:
            parsed_document: ParsedDocument to chunk

        Returns:
            ChunkingResult with list of chunks and statistics

        Raises:
            ValueError: If parsed_document.content is empty
        """
        if not parsed_document.content.strip():
            raise ValueError("Cannot chunk empty document")

        self.logger.info(f"Chunking document: {parsed_document.metadata.document}")

        chunks = self._create_chunks(parsed_document)

        result = ChunkingResult(
            document_path=parsed_document.metadata.document,
            chunks=chunks,
            total_chunks=len(chunks),
            total_characters=sum(c.character_count for c in chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )

        self.logger.info(
            f"Created {result.total_chunks} chunks from "
            f"{parsed_document.metadata.document} "
            f"({result.total_tokens} tokens)"
        )

        return result

    def _create_chunks(self, parsed_document: ParsedDocument) -> List[TextChunk]:
        """
        Internal method to create chunks from a parsed document.

        Args:
            parsed_document: ParsedDocument to chunk

        Returns:
            List of TextChunk objects
        """
        # Split text into semantic units (paragraphs first)
        semantic_units = self._split_into_semantic_units(parsed_document.content)

        # Group semantic units into chunks respecting size limits
        grouped_chunks = self._group_into_chunks(semantic_units)

        # Create TextChunk objects with metadata
        chunks: List[TextChunk] = []
        current_char_pos = 0

        for idx, chunk_text in enumerate(grouped_chunks):
            start_char = current_char_pos
            end_char = current_char_pos + len(chunk_text)

            chunk_metadata = ChunkMetadata(
                document=parsed_document.metadata.document,
                document_title=parsed_document.metadata.title,
                document_author=parsed_document.metadata.author,
                chunk_index=idx,
                total_chunks=len(grouped_chunks),
                start_char=start_char,
                end_char=end_char,
            )

            text_chunk = TextChunk(
                content=chunk_text,
                metadata=chunk_metadata,
            )

            chunks.append(text_chunk)
            current_char_pos = end_char

        return chunks

    def _split_into_semantic_units(self, text: str) -> List[str]:
        """
        Split text into semantic units (paragraphs, sentences).

        Priority order:
        1. Non-empty paragraphs (separated by blank lines)
        2. If paragraph too large, split by sentences
        3. If sentence too large, split by words (last resort)

        Args:
            text: Raw text to split

        Returns:
            List of semantic units
        """
        units = []

        # First, split by paragraphs (multiple newlines)
        paragraphs = re.split(r"\n\s*\n", text.strip())

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Calculate token count for this paragraph
            para_tokens = self._count_tokens(para)

            if para_tokens <= self.chunk_size:
                # Paragraph fits in one chunk
                units.append(para)
            else:
                # Paragraph is too large, split by sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_tokens = self._count_tokens(sentence)

                    if sentence_tokens <= self.chunk_size:
                        units.append(sentence)
                    else:
                        # Sentence is too large, split by words
                        words = sentence.split()
                        word_chunks = self._split_words_into_chunks(words)
                        units.extend(word_chunks)

        return units

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting: period, exclamation, question marks
        # Followed by space and capital letter or end of text
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Handle remaining sentences (those not followed by capital letter)
        result = []
        for sent in sentences:
            # Further split by newlines if present
            for sub_sent in sent.split("\n"):
                sub_sent = sub_sent.strip()
                if sub_sent:
                    result.append(sub_sent)

        return result

    def _split_words_into_chunks(self, words: List[str]) -> List[str]:
        """
        Split words into chunks respecting size limit.

        Args:
            words: List of words

        Returns:
            List of word-based chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            # Approximate: 1 token per 0.75 words = 1 word ≈ 1.33 tokens
            word_tokens = max(1, round(len(word.split()) / 0.75))

            if current_tokens + word_tokens > self.chunk_size and current_chunk:
                # Current chunk is full, save it
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _group_into_chunks(self, semantic_units: List[str]) -> List[str]:
        """
        Group semantic units into chunks with overlap.

        Uses a sliding window approach:
        - Window size: chunk_size
        - Overlap: overlap_size
        - Each window contains complete semantic units when possible

        Args:
            semantic_units: List of semantic units to group

        Returns:
            List of grouped chunks (text)
        """
        if not semantic_units:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for unit in semantic_units:
            unit_tokens = self._count_tokens(unit)

            # Check if adding this unit would exceed chunk size
            if current_tokens + unit_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                # Calculate how many units fit in overlap window
                overlap_tokens = 0
                overlap_units = []

                for u in reversed(current_chunk):
                    u_tokens = self._count_tokens(u)
                    if overlap_tokens + u_tokens <= self.overlap_size:
                        overlap_units.insert(0, u)
                        overlap_tokens += u_tokens
                    else:
                        break

                current_chunk = overlap_units + [unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current_chunk.append(unit)
                current_tokens += unit_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count for text.

        Uses heuristic: 1 token ≈ 0.75 words

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        word_count = len(text.split())
        return max(1, round(word_count / 0.75))
