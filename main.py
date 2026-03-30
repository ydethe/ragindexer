#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Indexer - Main Entry Point

Entry point for the real-time RAG indexing pipeline.
Loads configuration and starts the PipelineOrchestrator.

Configuration is loaded from Settings (environment variables and .env file).
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from ragindexer import Settings
from ragindexer.Orchestrator import PipelineOrchestrator

# Initialize rich console for formatting
console = Console()

# Configure root logger with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=False, console=console)],
)

logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main entry point for the RAG Indexer pipeline.

    Loads configuration from Settings and runs the orchestrator.

    Environment variables (via Settings):
    - DEBOUNCE_DELAY: Seconds to wait before syncing (default: 2.0)
    - ENV_FILE: Path to .env file (default: ".env")

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load settings from environment and .env file
        settings = Settings()

        # Create and run orchestrator
        orchestrator = PipelineOrchestrator(
            settings,
            debounce_delay=settings.DEBOUNCE_DELAY,
            console=console,
        )
        return orchestrator.run()

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
