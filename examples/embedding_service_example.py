# -*- coding: utf-8 -*-
"""
Example usage of EmbeddingService (Component 4)

Demonstrates:
1. Generating embeddings for document chunks
2. Calculating similarity between chunks
3. Full pipeline integration (scan → parse → chunk → embed)
4. Using embeddings for semantic search
"""

from pathlib import Path
from ragindexer import (
    FileScanner,
    DocumentParser,
    ChunkingService,
    EmbeddingService,
)


def example_1_basic_embedding():
    """Example 1: Basic embedding of chunks"""
    print("\n=== Example 1: Basic Embedding ===\n")

    # Create a sample document
    sample_dir = Path("./sample_docs")
    sample_dir.mkdir(exist_ok=True)

    sample_file = sample_dir / "example.txt"
    sample_file.write_text(
        """
    Machine learning is a subset of artificial intelligence.
    It enables systems to learn and improve from experience.

    Deep learning uses neural networks with multiple layers.
    It has revolutionized computer vision and natural language processing.

    Embeddings are numerical representations of text.
    They capture semantic meaning and enable similarity calculations.
    """
    )

    # Step 1: Scan for files
    print("Step 1: Scanning for files...")
    scanner = FileScanner(sample_dir)
    scan_result = scanner.scan()
    print(f"Found {scan_result.total_files} files")

    # Step 2: Parse documents
    print("\nStep 2: Parsing documents...")
    parser = DocumentParser()
    file_info = list(scan_result.files.values())[0]
    parsed_doc = parser.parse(file_info)
    print(f"Parsed content length: {parsed_doc.character_count} chars")

    # Step 3: Chunk the document
    print("\nStep 3: Chunking document...")
    chunking_service = ChunkingService(chunk_size=100, overlap_size=20)
    chunking_result = chunking_service.chunk(parsed_doc)
    print(f"Created {chunking_result.total_chunks} chunks")

    # Step 4: Embed chunks
    print("\nStep 4: Embedding chunks...")
    embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    embedding_result = embedding_service.embed_chunks(chunking_result.chunks)
    print(
        f"Embedded {embedding_result.total_chunks} chunks in {embedding_result.total_time_seconds:.2f}s"
    )
    print(f"Embedding dimension: {embedding_result.embedding_dim}")

    # Display results
    print("\nEmbedding Results:")
    for i, embedded_chunk in enumerate(embedding_result.embedded_chunks):
        print(f"\nChunk {i}:")
        print(f"  Content: {embedded_chunk.chunk.content[:50]}...")
        print(f"  Tokens: {embedded_chunk.chunk.token_count}")
        print(f"  Embedding (first 5 dims): {embedded_chunk.embedding[:5]}")

    return embedding_service, embedding_result


def example_2_similarity_calculation(embedding_service, embedding_result):
    """Example 2: Calculate similarity between chunks"""
    print("\n\n=== Example 2: Similarity Calculation ===\n")

    embedded_chunks = embedding_result.embedded_chunks

    if len(embedded_chunks) >= 2:
        # Compare first two chunks
        chunk1 = embedded_chunks[0]
        chunk2 = embedded_chunks[1]

        similarity = embedding_service.similarity(chunk1.embedding, chunk2.embedding)

        print(f"Chunk 1: {chunk1.chunk.content[:50]}...")
        print(f"Chunk 2: {chunk2.chunk.content[:50]}...")
        print(f"Similarity: {similarity:.4f}")

    # Find most similar chunks
    print("\nFinding most similar chunk pairs:")
    max_similarity = -1
    max_pair = None

    for i in range(len(embedded_chunks)):
        for j in range(i + 1, len(embedded_chunks)):
            similarity = embedding_service.similarity(
                embedded_chunks[i].embedding,
                embedded_chunks[j].embedding,
            )
            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (i, j)

    if max_pair:
        i, j = max_pair
        print(f"Most similar pair: Chunk {i} and {j} " f"(similarity: {max_similarity:.4f})")
        print(f"  Chunk {i}: {embedded_chunks[i].chunk.content[:40]}...")
        print(f"  Chunk {j}: {embedded_chunks[j].chunk.content[:40]}...")


def example_3_query_embedding(embedding_service):
    """Example 3: Embed a query and find similar chunks"""
    print("\n\n=== Example 3: Semantic Search ===\n")

    # Create sample query embeddings
    queries = [
        "What is machine learning?",
        "How do embeddings work?",
        "What is neural network?",
    ]

    print("Query embeddings (first 5 dimensions):")
    for query in queries:
        query_embedding = embedding_service.embed_text(query)
        print(f"Query: '{query}'")
        print(f"  Embedding: {query_embedding[:5]}...")
        print(f"  Vector length: {len(query_embedding)}")


def example_4_model_caching():
    """Example 4: Demonstrate model caching"""
    print("\n\n=== Example 4: Model Caching ===\n")

    import time

    # First instantiation (loads model)
    print("Creating EmbeddingService #1...")
    start = time.time()
    service1 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    time1 = time.time() - start
    print(f"Time: {time1:.3f}s")

    # Second instantiation (uses cache)
    print("\nCreating EmbeddingService #2...")
    start = time.time()
    service2 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    time2 = time.time() - start
    print(f"Time: {time2:.3f}s")

    # Verify they use the same model
    print(f"\nSame model instance? {service1.model is service2.model}")
    if time2 > 0:
        print(f"Speedup: {time1/time2:.1f}x faster (from cache)")
    else:
        print("Cache is effective (second load < 1ms)")


def example_5_batch_processing():
    """Example 5: Batch processing performance"""
    print("\n\n=== Example 5: Batch Processing ===\n")

    from ragindexer import TextChunk, ChunkMetadata

    # Create test chunks
    test_chunks = []
    for i in range(10):
        metadata = ChunkMetadata(
            document="test.txt",
            chunk_index=i,
            total_chunks=10,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        chunk = TextChunk(
            content=f"Sample chunk {i} with some meaningful content about machine learning and AI topics.",
            metadata=metadata,
        )
        test_chunks.append(chunk)

    # Embed with different batch sizes
    for batch_size in [1, 5, 10]:
        import time

        embedding_service = EmbeddingService(
            batch_size=batch_size, model_name="BAAI/bge-small-en-v1.5"
        )

        start = time.time()
        result = embedding_service.embed_chunks(test_chunks)
        elapsed = time.time() - start

        print(f"Batch size: {batch_size}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {result.total_chunks / elapsed:.1f} chunks/s")

        embedding_service.clear_cache()


def main():
    """Run all examples"""
    print("EmbeddingService Examples")
    print("=" * 50)

    # Example 1: Basic embedding
    embedding_service, embedding_result = example_1_basic_embedding()

    # Example 2: Similarity calculation
    example_2_similarity_calculation(embedding_service, embedding_result)

    # Example 3: Query embedding
    example_3_query_embedding(embedding_service)

    # Example 4: Model caching
    example_4_model_caching()

    # Example 5: Batch processing
    example_5_batch_processing()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
