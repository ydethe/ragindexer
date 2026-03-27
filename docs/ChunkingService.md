# ChunkingService Documentation

## Overview

The **ChunkingService** component splits parsed documents into semantically coherent chunks optimized for embedding generation. It handles intelligent chunking with overlap, preserving semantic boundaries (paragraphs, sentences) and associating rich metadata with each chunk.

## Purpose

After extracting text from documents using the `DocumentParser`, text must be split into manageable pieces for embedding generation. Chunking serves multiple purposes:

1. **Manageable Size**: Embeddings have token limits; chunking ensures each piece is within bounds
2. **Semantic Coherence**: Preserves meaningful boundaries (paragraphs, sentences) for better embeddings
3. **Continuity**: Overlap between chunks maintains context across boundaries
4. **Metadata Association**: Each chunk retains information about its source and position

## Architecture

### Class: `ChunkingService`

Main service for chunking operations.

```python
ChunkingService(
    chunk_size: int = 512,        # Target chunk size in tokens
    overlap_size: int = 50,       # Overlap between chunks in tokens
    logger_instance: Optional[logging.Logger] = None
)
```

**Parameters:**
- `chunk_size`: Target size of chunks in tokens (typically 256-1024)
- `overlap_size`: Number of tokens to overlap between consecutive chunks (typically 50-200)
- `logger_instance`: Optional logger instance (defaults to module logger)

### Main Method: `chunk()`

```python
def chunk(parsed_document: ParsedDocument) -> ChunkingResult
```

Splits a `ParsedDocument` into `TextChunk` objects.

**Input:**
- `ParsedDocument`: Document parsed by `DocumentParser`

**Output:**
- `ChunkingResult`: Result object containing:
  - List of `TextChunk` objects
  - Total chunks, characters, and tokens
  - Chunking timestamp

**Raises:**
- `ValueError`: If document content is empty

## Data Models

### TextChunk

Represents a single chunk of text ready for embedding.

```python
class TextChunk(BaseModel):
    content: str                      # Actual text content
    metadata: ChunkMetadata           # Associated metadata
    character_count: int              # Character count (auto-calculated)
    token_count: int                  # Token count (auto-calculated)
```

### ChunkMetadata

Metadata associated with a chunk.

```python
class ChunkMetadata(BaseModel):
    source_file: str                  # Original file path
    document_title: Optional[str]     # Document title (if available)
    document_author: Optional[str]    # Document author (if available)
    chunk_index: int                  # Sequential chunk index
    total_chunks: int                 # Total chunks in document
    start_char: int                   # Character position in original text
    end_char: int                     # Character position in original text
    extracted_at: datetime            # When chunk was created
```

### ChunkingResult

Result of chunking a document.

```python
class ChunkingResult(BaseModel):
    document_path: str                # Source document path
    chunks: List[TextChunk]          # List of created chunks
    total_chunks: int                 # Total number of chunks
    total_characters: int             # Total characters processed
    total_tokens: int                 # Total tokens across all chunks
    chunking_time: datetime           # When chunking was performed
```

## Chunking Strategy

The service employs an intelligent multi-level chunking strategy:

### Level 1: Paragraph Split
- Split text by blank lines (multiple newlines)
- Each paragraph becomes a candidate unit
- Preserves document structure

### Level 2: Sentence Split
- If a paragraph exceeds `chunk_size`, split by sentences
- Sentences are identified by `.`, `!`, `?` followed by space and capital letter
- Handles multiple newlines within paragraphs

### Level 3: Word Split
- If a sentence exceeds `chunk_size`, split by individual words
- Last resort to ensure no unit is larger than limit
- Reconstructs chunks by concatenating words

### Level 4: Grouping with Overlap
- Groups semantic units into chunks respecting size limits
- Implements sliding window overlap:
  - Window size: `chunk_size`
  - Overlap: `overlap_size` tokens from previous chunk
- Maintains context continuity across chunks

## Token Counting

The service approximates token count using a simple heuristic:

```
token_count = round(word_count / 0.75)
```

This is based on the observation that:
- 1 token ≈ 0.75 words (on average)
- 1 word ≈ 1.33 tokens

This approximation is suitable for:
- Budgeting token limits
- Controlling chunk sizes
- Performance estimation

**Note:** Actual token counts depend on tokenizer used (subword tokenization, BPE, etc.)

## Usage Example

```python
from ragindexer import FileScanner, DocumentParser, ChunkingService

# 1. Scan documents
scanner = FileScanner("/path/to/documents")
scan_result = scanner.scan()

# 2. Parse a document
parser = DocumentParser()
file_info = scan_result.files["document.txt"]
parsed_doc = parser.parse(file_info)

# 3. Chunk it
chunking_service = ChunkingService(
    chunk_size=512,      # tokens
    overlap_size=50      # tokens
)
chunking_result = chunking_service.chunk(parsed_doc)

# 4. Process chunks
for chunk in chunking_result.chunks:
    print(f"Chunk {chunk.metadata.chunk_index}:")
    print(f"  Content: {chunk.content}")
    print(f"  Tokens: {chunk.token_count}")
    print(f"  Position: {chunk.metadata.start_char}-{chunk.metadata.end_char}")
```

## Configuration

Chunking parameters should be chosen based on:

1. **Embedding Model**
   - Different models have different optimal chunk sizes
   - Sentence-BERT: 256-512 tokens
   - GPT embeddings: 512-2048 tokens

2. **Search Quality vs Granularity Trade-off**
   - Smaller chunks: More granular search, more embedding vectors
   - Larger chunks: Less granular, fewer vectors, faster search

3. **Overlap Strategy**
   - Larger overlap: Better continuity, more redundancy
   - Smaller overlap: Less redundancy, cheaper storage
   - Typical: 50-200 tokens

**Recommended defaults:**
```python
chunking_service = ChunkingService(
    chunk_size=512,      # Good balance for most models
    overlap_size=50      # Provides context continuity
)
```

## Performance Considerations

- **Memory:** Processes one document at a time; minimal memory impact
- **Speed:** Sub-second for typical documents (< 100KB)
- **Scalability:** Linear with document size and number of chunks

## Integration with Other Components

### With DocumentParser
```python
parser = DocumentParser()
parsed_doc = parser.parse(file_info)
chunking = ChunkingService()
chunking_result = chunking.chunk(parsed_doc)
```

### With EmbeddingService (future)
```python
# Next component in pipeline
embedder = EmbeddingService()
for chunk in chunking_result.chunks:
    embedding = embedder.embed(chunk.content)
    # Store embedding + metadata
```

### Data Flow
```
FileScanner (files)
  ↓
DocumentParser (parsed_doc: ParsedDocument)
  ↓
ChunkingService (chunking_result: ChunkingResult)
  ↓
[Future] EmbeddingService (embeddings)
  ↓
[Future] Vector Database (indexed)
```

## Testing

See `tests/test_chunking_service.py` for comprehensive test coverage (96% coverage):

- Basic chunking functionality
- Character counting and position tracking
- Token counting approximation
- Semantic preservation (paragraphs, sentences)
- Overlap functionality
- Result statistics
- Integration with DocumentParser

**Run tests:**
```bash
./.venv/Scripts/python.exe -m pytest tests/test_chunking_service.py -v
```

## Error Handling

The service raises specific errors:

```python
# Empty document
chunking.chunk(empty_doc)  # ValueError: "Cannot chunk empty document"
```

All errors are logged with context for debugging.

## API Reference

### ChunkingService Methods

```python
def chunk(parsed_document: ParsedDocument) -> ChunkingResult
    """Split a parsed document into chunks."""

def _create_chunks(parsed_document: ParsedDocument) -> List[TextChunk]
    """Internal: Create TextChunk objects with metadata."""

def _split_into_semantic_units(text: str) -> List[str]
    """Internal: Split text into paragraphs, sentences, or words."""

def _split_into_sentences(text: str) -> List[str]
    """Internal: Split text by sentence boundaries."""

def _split_words_into_chunks(words: List[str]) -> List[str]
    """Internal: Split words into token-sized chunks."""

def _group_into_chunks(semantic_units: List[str]) -> List[str]
    """Internal: Group units into chunks with overlap."""

def _count_tokens(text: str) -> int
    """Internal: Approximate token count for text."""
```

## Future Enhancements

Potential improvements to consider:

1. **Custom Tokenizer Support**: Use actual tokenizer (e.g., tiktoken, transformers) instead of word approximation
2. **Semantic Chunking**: Use embeddings to find optimal chunk boundaries
3. **Hierarchical Chunking**: Create chunk hierarchy (document → sections → paragraphs → sentences)
4. **Parallel Processing**: Process multiple documents concurrently
5. **Adaptive Sizing**: Adjust chunk size based on document type or language
6. **Sliding Sentence Window**: More sophisticated sentence overlap strategy

## Related Files

- `src/ragindexer/ChunkingService.py` - Implementation
- `tests/test_chunking_service.py` - Test suite
- `examples/chunking_service_example.py` - Usage examples
- `Architecture.md` - System architecture
