# -*- coding: utf-8 -*-
"""
Example usage of VectorDatabaseService (Component 5)

Demonstrates:
1. Storing embeddings in the vector database
2. Semantic search
3. Document management
4. Full pipeline integration (scan → parse → chunk → embed → store)
"""

from pathlib import Path
from ragindexer import (
    FileScanner,
    DocumentParser,
    ChunkingService,
    EmbeddingService,
    VectorDatabaseService,
)


def example_1_basic_storage():
    """Example 1: Basic embedding storage and retrieval."""
    print("\n=== Example 1: Basic Storage and Search ===\n")

    # Create sample documents
    sample_dir = Path("./sample_docs_db")
    sample_dir.mkdir(exist_ok=True)

    (sample_dir / "document1.txt").write_text(
        """Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without being explicitly programmed.

Deep learning uses artificial neural networks with multiple layers.
It has achieved breakthrough results in computer vision and natural language processing.

Embeddings are vector representations of text that capture semantic meaning.
They allow us to perform similarity calculations between documents."""
    )

    (sample_dir / "document2.txt").write_text(
        """Python is a popular programming language for machine learning.
Libraries like TensorFlow and PyTorch make it easy to build neural networks.

Data science involves collecting, processing, and analyzing large datasets.
Statistical analysis and visualization are key techniques in data science."""
    )

    # Setup services
    scanner = FileScanner(sample_dir)
    parser = DocumentParser()
    chunking_service = ChunkingService(chunk_size=150, overlap_size=30)
    embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

    # Use persistent storage
    db_path = Path("./data_db/qdrant")
    vector_db = VectorDatabaseService(
        vector_size=384,
        persistence_path=db_path,
    )

    # Index documents
    print("Step 1: Scanning documents...")
    scan_result = scanner.scan()
    print(f"Found {scan_result.total_files} files")

    print("\nStep 2: Parsing, chunking, and embedding...")
    total_embeddings = 0

    for file_info in scan_result.files.values():
        print(f"  Processing: {file_info.relative_path}")

        # Parse
        parsed_doc = parser.parse(file_info)

        # Chunk
        chunking_result = chunking_service.chunk(parsed_doc)

        # Embed
        embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

        # Store
        db_result = vector_db.add_embeddings(embedding_result.embedded_chunks)

        if db_result.success:
            total_embeddings += db_result.items_affected
            print(
                f"    ✓ Stored {db_result.items_affected} chunks "
                f"({db_result.duration_seconds:.2f}s)"
            )
        else:
            print(f"    ✗ Error: {db_result.error}")

    # Show database statistics
    print("\nStep 3: Database Statistics")
    stats = vector_db.get_statistics()
    print(f"  Total embeddings: {stats['point_count']}")
    print(f"  Vector dimensions: {stats['vector_size']}")
    print(f"  Collection: {stats['collection_name']}")

    return vector_db, embedding_service


def example_2_semantic_search(vector_db, embedding_service):
    """Example 2: Perform semantic searches."""
    print("\n\n=== Example 2: Semantic Search ===\n")

    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is data science?",
        "Python programming libraries",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        print("  Results:")

        # Generate query embedding
        query_embedding = embedding_service.embed_text(query)

        # Search
        search_result = vector_db.search(
            query_embedding.tolist(),
            limit=3,
            score_threshold=0.5,
        )

        if search_result.success and search_result.results:
            for i, result in enumerate(search_result.results, 1):
                print(f"    {i}. [{result.score:.3f}] {result.source_file}")
                print(f"       {result.chunk_content[:70]}...")
        else:
            print("    No results found")

        print()


def example_3_document_management(vector_db):
    """Example 3: Document deletion and management."""
    print("\n=== Example 3: Document Management ===\n")

    # Show initial count
    stats_before = vector_db.get_statistics()
    print(f"Total embeddings before: {stats_before['point_count']}")

    # Delete a document
    doc_to_delete = "sample_docs_db/document1.txt"
    print(f"\nDeleting: {doc_to_delete}")

    result = vector_db.delete_document(doc_to_delete)

    if result.success:
        print(f"Deleted {result.items_affected} embeddings")

        # Show new count
        stats_after = vector_db.get_statistics()
        print(f"Total embeddings after: {stats_after['point_count']}")
    else:
        print(f"Error: {result.error}")


def example_4_batch_operations():
    """Example 4: Batch operations for performance."""
    print("\n\n=== Example 4: Batch Operations ===\n")

    import time

    # Create vector database
    vector_db = VectorDatabaseService(vector_size=384, persistence_path=None)

    # Create many fake embeddings
    from ragindexer import TextChunk, ChunkMetadata, EmbeddedChunk

    print("Creating 100 sample embeddings...")
    chunks = []

    for i in range(100):
        metadata = ChunkMetadata(
            source_file=f"batch_test_{i // 10}.txt",
            chunk_index=i % 10,
            total_chunks=10,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        chunk = TextChunk(
            content=f"Sample text content {i}",
            metadata=metadata,
        )
        # Simple embedding
        embedding = [0.1 * ((i + j) % 384) for j in range(384)]

        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_dim=384,
            embedding_model="test",
        )
        chunks.append(embedded_chunk)

    # Add in batch
    print("Adding to database...")
    start = time.time()
    result = vector_db.add_embeddings(chunks)
    elapsed = time.time() - start

    print(f"✓ Added {result.items_affected} embeddings in {elapsed:.2f}s")
    print(f"  Speed: {result.items_affected / elapsed:.0f} embeddings/sec")

    # Stats
    stats = vector_db.get_statistics()
    print(f"  Total in database: {stats['point_count']}")


def example_5_persistence():
    """Example 5: Demonstrate persistence across instances."""
    print("\n\n=== Example 5: Persistence ===\n")

    db_path = Path("./data_db/persistence_test")

    # First instance
    print("Creating instance 1...")
    vector_db1 = VectorDatabaseService(
        persistence_path=db_path,
        collection_name="test_collection",
    )

    stats1 = vector_db1.get_statistics()
    print(f"Instance 1 - Total embeddings: {stats1['point_count']}")

    # Create a new instance with same path
    print("\nCreating instance 2 with same path...")
    vector_db2 = VectorDatabaseService(
        persistence_path=db_path,
        collection_name="test_collection",
    )

    stats2 = vector_db2.get_statistics()
    print(f"Instance 2 - Total embeddings: {stats2['point_count']}")

    print(f"\n✓ Same data persisted: {stats1['point_count'] == stats2['point_count']}")


def main():
    """Run all examples."""
    print("VectorDatabaseService Examples")
    print("=" * 60)

    # Example 1
    vector_db, embedding_service = example_1_basic_storage()

    # Example 2
    example_2_semantic_search(vector_db, embedding_service)

    # Example 3
    example_3_document_management(vector_db)

    # Example 4
    example_4_batch_operations()

    # Example 5
    example_5_persistence()

    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()
