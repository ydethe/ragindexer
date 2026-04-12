#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocumentParser Usage Example

This example demonstrates:
1. Scanning a folder with FileScanner
2. Parsing detected documents with DocumentParser
3. Processing extracted content
4. Error handling
"""

import logging
from pathlib import Path
from ragindexer.FileScanner import FileScanner
from ragindexer.DocumentParser import DocumentParser

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """
    Main example: scan documents and parse them.
    """
    # Path to documents folder
    docs_folder = Path("./sample_docs")

    # You would typically use a real folder with documents
    # For this example, create one with test documents
    if not docs_folder.exists():
        print(f"Creating sample documents in {docs_folder}...")
        docs_folder.mkdir(exist_ok=True)

        # Create sample files for demonstration
        (docs_folder / "sample.txt").write_text(
            "This is a sample text file.\nIt has multiple lines.\nVery useful!",
            encoding="utf-8",
        )

        (docs_folder / "sample.md").write_text(
            "# Sample Markdown\n\nThis is a **markdown** file.\n\n## Section\n\nWith content.",
            encoding="utf-8",
        )

        print("Sample documents created.\n")

    # ============================================================================
    # STEP 1: Scan documents using FileScanner
    # ============================================================================
    print("=" * 70)
    print("STEP 1: Scanning documents with FileScanner")
    print("=" * 70)

    try:
        scanner = FileScanner(docs_folder)
        scan_result = scanner.scan()

        print(f"\nFound {scan_result.total_files} documents")
        print(f"Total size: {scan_result.total_size_bytes / 1024:.2f} KB\n")

        if scan_result.total_files == 0:
            print("No documents found. Exiting.")
            return

    except ValueError as e:
        print(f"Error scanning folder: {e}")
        return

    # ============================================================================
    # STEP 2: Parse documents using DocumentParser
    # ============================================================================
    print("=" * 70)
    print("STEP 2: Parsing documents with DocumentParser")
    print("=" * 70 + "\n")

    parser = DocumentParser()
    parsed_documents = []

    for rel_path, file_info in scan_result.files.items():
        print(f"Processing: {rel_path}")
        print(f"  Format: {file_info.format.value}")
        print(f"  Size: {file_info.file_size} bytes")

        try:
            # Parse the document
            parsed_doc = parser.parse(file_info)

            # Store for later use
            parsed_documents.append(parsed_doc)

            # Display extracted information
            print(f"  [OK] Extracted: {parsed_doc.character_count} characters")
            if parsed_doc.metadata.title:
                print(f"  Title: {parsed_doc.metadata.title}")
            if parsed_doc.metadata.author:
                print(f"  Author: {parsed_doc.metadata.author}")
            if parsed_doc.metadata.page_count:
                print(f"  Pages: {parsed_doc.metadata.page_count}")

            # Preview content (first 200 chars)
            preview = parsed_doc.content[:200].replace("\n", " ")
            print(f"  Preview: {preview}...\n")

        except IOError as e:
            print(f"  [ERROR] File access error: {e}\n")
        except Exception as e:
            print(f"  [ERROR] Parsing error: {e}\n")

    # ============================================================================
    # STEP 3: Display summary statistics
    # ============================================================================
    print("=" * 70)
    print("STEP 3: Summary Statistics")
    print("=" * 70 + "\n")

    if parsed_documents:
        total_chars = sum(doc.character_count for doc in parsed_documents)
        avg_chars = total_chars / len(parsed_documents)

        print(f"Documents parsed successfully: {len(parsed_documents)}")
        print(f"Total characters extracted: {total_chars:,}")
        print(f"Average chars per document: {avg_chars:,.0f}\n")

        # Format breakdown
        formats = {}
        for doc in parsed_documents:
            fmt = doc.metadata.format.value
            formats[fmt] = formats.get(fmt, 0) + 1

        print("Format breakdown:")
        for fmt, count in sorted(formats.items()):
            print(f"  - {fmt.upper()}: {count} document(s)")

    # ============================================================================
    # STEP 4: Example - Access parsed content for further processing
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Using Parsed Content")
    print("=" * 70 + "\n")

    if parsed_documents:
        # Example: Find documents containing specific keywords
        keyword = "sample"

        print(f'Searching for documents containing "{keyword}":\n')

        for doc in parsed_documents:
            if keyword.lower() in doc.content.lower():
                print(f"  [FOUND] {doc.metadata.document}")
                # You could count occurrences
                count = doc.content.lower().count(keyword.lower())
                print(f"    Occurrences: {count}\n")

    # ============================================================================
    # STEP 5: Typical next step - Chunking
    # ============================================================================
    print("=" * 70)
    print("STEP 5: Next Steps")
    print("=" * 70)
    print(
        """
The extracted ParsedDocument objects are ready for:

1. Chunking Service
   - Split content into semantic chunks
   - Maintain overlap for context
   - Preserve metadata per chunk

2. Embedding Service
   - Generate vectors for each chunk
   - Use a local model (sentence-transformers)

3. Vector Database
   - Store embeddings with metadata
   - Enable semantic search

4. Sync Manager
   - Orchestrate full pipeline
   - Handle incremental updates
   - Manage document deletions
"""
    )


if __name__ == "__main__":
    main()
