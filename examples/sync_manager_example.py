#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using SyncManager for Document Indexing

This example demonstrates:
1. Initializing a SyncManager
2. Performing initial full indexing
3. Handling incremental updates
4. Querying the indexed documents
"""

from pathlib import Path
from ragindexer import (
    SyncManager,
    SyncStatus,
    EmbeddingService,
)


def main():
    """Main example function"""

    # 1. Setup
    print("=" * 60)
    print("RAGIndexer - SyncManager Example")
    print("=" * 60)

    # Define document source and database paths
    doc_root = Path("./documents")
    db_path = Path("./data/qdrant")

    # Create SyncManager
    print("\n1️⃣  Initializing SyncManager...")
    sync_manager = SyncManager(
        scan_root=doc_root,
        persistence_path=db_path,
        chunk_size=512,
        overlap_size=50,
        embedding_model="all-MiniLM-L6-v2",
    )
    print(f"   ✅ SyncManager initialized")
    print(f"   📁 Document root: {doc_root}")
    print(f"   💾 Database path: {db_path}")

    # 2. Full initial indexing
    print("\n2️⃣  Performing full initial indexing...")
    result = sync_manager.full_sync()

    print(f"\n   📊 Indexing Results:")
    print(f"   - Overall status: {result.overall_status.value}")
    print(f"   - Files processed: {result.total_files_processed}")
    print(f"   - Files added: {result.total_files_added}")
    print(f"   - Total chunks: {result.total_chunks_created}")
    print(f"   - Errors: {result.total_errors}")
    print(f"   - Duration: {result.duration_seconds:.2f}s")

    # Show per-file results
    if result.file_results:
        print(f"\n   📄 Per-file Details:")
        for rel_path, file_result in result.file_results.items():
            status_emoji = "✅" if file_result.status == SyncStatus.COMPLETED else "❌"
            print(f"   {status_emoji} {rel_path}")
            print(f"      - Status: {file_result.status.value}")
            print(f"      - Chunks: {file_result.chunks_count}")
            print(f"      - Duration: {file_result.duration_seconds:.2f}s")
            if file_result.error:
                print(f"      - Error: {file_result.error}")

    # 3. Get database statistics
    print("\n3️⃣  Database Statistics:")
    stats = sync_manager.get_statistics()
    print(f"   - Collection: {stats.get('collection_name', 'N/A')}")
    print(f"   - Embeddings stored: {stats.get('point_count', 0)}")
    print(f"   - Vector dimension: {stats.get('vector_size', 0)}")
    print(f"   - Persistent storage: {stats.get('persistence', False)}")

    # 4. Example: Simulate changes and incremental sync
    print("\n4️⃣  Incremental Sync (simulated scenario):")
    print("   Assuming document changes occurred:")
    print("   - Modified: document.txt")
    print("   - Added: new_document.md")
    print("   - Deleted: old_file.txt")

    # In a real scenario, files would be changed on disk, then:
    # result = sync_manager.incremental_sync()
    # For this example, we just show the structure

    # 5. Example: Using embeddings for search
    print("\n5️⃣  Example: Semantic Search Setup:")
    print("   To use the indexed documents for semantic search:")

    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

    # Example search query
    query_text = "What is machine learning?"
    print(f"\n   Query: '{query_text}'")
    print("\n   Steps to search:")
    print("   1. Convert query to embedding:")
    query_result = embedding_service.embed_text(query_text)
    print(f"      ✓ Query embedding generated (dim: {len(query_result)})")

    print("   2. Search in vector database:")
    print("      search_results = sync_manager.vector_db.search(query_result)")

    # 6. Summary
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nKey SyncManager Methods:")
    print("  • full_sync()         - Initial complete indexing")
    print("  • incremental_sync()  - Process only changed files")
    print("  • get_statistics()    - Database statistics")
    print("  • get_last_scan_result() - Last scan information")
    print("\nFor more information, see docs/SyncManager.md")


if __name__ == "__main__":
    main()
