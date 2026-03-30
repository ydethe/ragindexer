#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChunkingService Usage Example

Demonstrates how to use the ChunkingService to split parsed documents
into semantically coherent chunks optimized for embedding generation.

Pipeline: FileScanner → DocumentParser → ChunkingService
"""

from pathlib import Path
from ragindexer import (
    FileScanner,
    DocumentParser,
    ChunkingService,
)


def main():
    """
    Example: Index a folder with chunking.

    1. Scan folder for documents
    2. Parse each document
    3. Chunk the parsed content
    """

    # Configuration
    documents_folder = Path("/path/to/documents")  # Change to your folder
    chunk_size = 512  # tokens
    overlap_size = 50  # tokens

    # Step 1: Scan the folder for documents
    print("=" * 60)
    print("STEP 1: Scanning documents folder")
    print("=" * 60)

    scanner = FileScanner(documents_folder)
    scan_result = scanner.scan()

    print(f"\n✓ Found {len(scan_result.files)} documents:")
    for file_path, file_info in scan_result.files.items():
        print(f"  - {file_path} ({file_info.file_size} bytes, {file_info.format.value})")

    # Step 2: Parse documents
    print("\n" + "=" * 60)
    print("STEP 2: Parsing documents")
    print("=" * 60)

    parser = DocumentParser()
    document_parser_results = {}

    for file_path, file_info in scan_result.files.items():
        try:
            parsed_doc = parser.parse(file_info)
            document_parser_results[file_path] = parsed_doc

            print(
                f"\n✓ Parsed: {file_path}"
                f"\n  - Title: {parsed_doc.metadata.title}"
                f"\n  - Author: {parsed_doc.metadata.author}"
                f"\n  - Characters: {parsed_doc.character_count}"
            )
        except Exception as e:
            print(f"\n✗ Failed to parse {file_path}: {e}")

    # Step 3: Chunk parsed documents
    print("\n" + "=" * 60)
    print("STEP 3: Chunking documents")
    print("=" * 60)

    chunking_service = ChunkingService(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    all_chunks = {}

    for file_path, parsed_doc in document_parser_results.items():
        try:
            chunking_result = chunking_service.chunk(parsed_doc)
            all_chunks[file_path] = chunking_result

            print(f"\n✓ Chunked: {file_path}")
            print(f"  - Total chunks: {chunking_result.total_chunks}")
            print(f"  - Total tokens: {chunking_result.total_tokens}")
            print(f"  - Total characters: {chunking_result.total_characters}")

            # Show first 2 chunks as example
            print(f"\n  First chunk:")
            if chunking_result.chunks:
                first_chunk = chunking_result.chunks[0]
                preview = (
                    first_chunk.content[:100] + "..."
                    if len(first_chunk.content) > 100
                    else first_chunk.content
                )
                print(f"    Content: {preview}")
                print(f"    Tokens: {first_chunk.token_count}")
                print(
                    f"    Position: {first_chunk.metadata.start_char}-"
                    f"{first_chunk.metadata.end_char}"
                )

        except Exception as e:
            print(f"\n✗ Failed to chunk {file_path}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_chunks = sum(result.total_chunks for result in all_chunks.values())
    total_tokens = sum(result.total_tokens for result in all_chunks.values())

    print(f"\n✓ Total documents processed: {len(all_chunks)}")
    print(f"✓ Total chunks created: {total_chunks}")
    print(f"✓ Total tokens: {total_tokens}")

    print("\n" + "=" * 60)
    print("Ready for embedding generation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
