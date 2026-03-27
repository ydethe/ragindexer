# -*- coding: utf-8 -*-
"""
RAG Indexer - Pipeline Orchestrator

Orchestrates the real-time RAG indexing pipeline with:
- Filesystem event monitoring with debouncing
- Initial full sync and incremental syncs
- Graceful shutdown handling
- Progress reporting and result display
"""

import logging
import signal
import threading
import time
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ragindexer import Settings, SyncManager

logger = logging.getLogger(__name__)


class SyncEventHandler(FileSystemEventHandler):
    """
    Filesystem event handler that triggers syncs on file changes.

    Features:
    - Debouncing to avoid rapid repeated syncs for batch operations
    - Configurable debounce delay
    - Hook callbacks for file events
    """

    def __init__(self, on_change_hook, debounce_delay: float = 2.0):
        """
        Initialize the event handler.

        Args:
            on_change_hook: Callable that gets invoked on filesystem changes
            debounce_delay: Seconds to wait before triggering sync (default: 2.0s)
        """
        super().__init__()
        self.on_change_hook = on_change_hook
        self.debounce_delay = debounce_delay
        self.debounce_timer = None
        self.lock = threading.Lock()

    def _trigger_sync(self):
        """Execute the sync hook."""
        try:
            self.on_change_hook()
        except Exception as e:
            logger.error(f"Error in sync hook: {e}")

    def _schedule_sync(self):
        """Schedule sync with debouncing."""
        with self.lock:
            # Cancel existing timer if any
            if self.debounce_timer is not None:
                self.debounce_timer.cancel()

            # Schedule new sync after debounce delay
            self.debounce_timer = threading.Timer(self.debounce_delay, self._trigger_sync)
            self.debounce_timer.daemon = True
            self.debounce_timer.start()

    def on_created(self, event):
        """Hook called when a file or directory is created."""
        if not event.is_directory:
            logger.debug(f"📝 File created: {event.src_path}")
            self._schedule_sync()

    def on_modified(self, event):
        """Hook called when a file or directory is modified."""
        if not event.is_directory:
            logger.debug(f"✏️  File modified: {event.src_path}")
            self._schedule_sync()

    def on_deleted(self, event):
        """Hook called when a file or directory is deleted."""
        if not event.is_directory:
            logger.debug(f"🗑️  File deleted: {event.src_path}")
            self._schedule_sync()


class PipelineOrchestrator:
    """
    Orchestrates the real-time RAG indexing pipeline.

    Features:
    - Initial full synchronization of all documents
    - Real-time monitoring of root folder for changes
    - Graceful shutdown on signals (CTRL+C)
    - Detailed logging and progress tracking
    """

    def __init__(
        self,
        settings: Settings,
        debounce_delay: float = 2.0,
        console: Console = None,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            settings: Settings object with configuration
            debounce_delay: Seconds to wait before syncing on filesystem changes
            console: Rich Console instance for output (creates new if not provided)
        """
        self.settings = settings
        self.logger = logger
        self.running = True
        self.debounce_delay = debounce_delay
        self.console = console or Console()
        self._setup_signal_handlers()

        # Initialize SyncManager with settings
        self.sync_manager = SyncManager(
            scan_root=settings.get_scan_root(),
            persistence_path=settings.get_qdrant_persistence_path(),
            chunk_size=settings.CHUNK_SIZE,
            overlap_size=settings.OVERLAP_SIZE,
            embedding_model=settings.EMBEDDING_MODEL,
            logger_instance=self.logger,
        )

        # Initialize filesystem observer
        self.observer = Observer()
        self.event_handler = SyncEventHandler(
            on_change_hook=self._on_filesystem_change,
            debounce_delay=debounce_delay,
        )

        self.logger.info("🚀 RAG Indexer Pipeline initialized")
        self.logger.info(f"📁 Monitoring: {settings.get_scan_root()}")
        self.logger.info(f"🔤 Embedding model: {settings.EMBEDDING_MODEL}")
        self.logger.info(
            f"📏 Chunk size: {settings.CHUNK_SIZE} tokens, overlap: {settings.OVERLAP_SIZE}"
        )
        self.logger.info(f"⏱️  Debounce delay: {debounce_delay}s")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info("\n⏹️  Shutdown signal received. Stopping gracefully...")
            self.running = False
            self.observer.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _on_filesystem_change(self) -> None:
        """Hook called when filesystem changes are detected."""
        if not self.running:
            return

        self.logger.info(f"\n🔄 Filesystem change detected [{datetime.now().strftime('%H:%M:%S')}]")

        try:
            result = self.sync_manager.incremental_sync()

            # Only display if there were changes
            if result.total_files_processed > 0:
                self.logger.info("📊 Changes detected:")
                self._display_sync_result(result)
            else:
                self.logger.info("✅ No changes detected")

        except ValueError:
            # Should not happen as full_sync was already called
            self.logger.error("Error: No previous scan result. Stopping.")
            self.running = False
        except Exception as e:
            self.logger.error(f"❌ Incremental sync error: {e}")

    def _display_sync_result(self, result) -> None:
        """Display sync operation results in a formatted table."""
        table = Table(title="Sync Operation Result", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", str(result.overall_status).upper())
        table.add_row("Files processed", str(result.total_files_processed))
        table.add_row("Files added", str(result.total_files_added))
        table.add_row("Files modified", str(result.total_files_modified))
        table.add_row("Files deleted", str(result.total_files_deleted))
        table.add_row("Total chunks", str(result.total_chunks_created))
        table.add_row("Errors", str(result.total_errors))
        table.add_row("Duration", f"{result.duration_seconds:.2f}s")

        self.console.print(table)

        # Display any errors
        if result.file_results:
            error_results = {
                path: res
                for path, res in result.file_results.items()
                if res.status.value == "failed"
            }
            if error_results:
                error_table = Table(title="Failed Files", show_header=True, header_style="bold red")
                error_table.add_column("File", style="cyan")
                error_table.add_column("Error", style="red")

                for path, result in error_results.items():
                    error_table.add_row(path, result.error or "Unknown error")

                self.console.print(error_table)

    def run_initial_sync(self) -> bool:
        """
        Perform initial full synchronization of all documents.

        Returns:
            True if sync succeeded (fully or partially), False if completely failed
        """
        self.logger.info("=" * 60)
        self.logger.info("📑 STARTING INITIAL FULL SYNC")
        self.logger.info("=" * 60)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                progress.add_task(description="Performing initial sync...", total=None)
                result = self.sync_manager.full_sync()

            self._display_sync_result(result)

            if result.overall_status.value != "failed":
                self.logger.info("✅ Initial sync completed successfully")
                return True
            else:
                self.logger.error("❌ Initial sync failed completely")
                return False

        except Exception as e:
            self.logger.error(f"❌ Initial sync failed with error: {e}")
            return False

    def run_monitoring_loop(self) -> None:
        """
        Run the filesystem event-driven monitoring.

        Watches the root folder for changes and performs incremental syncs
        when filesystem events are detected.
        """
        self.logger.info("=" * 60)
        self.logger.info("👁️  ENTERING EVENT-DRIVEN MONITORING MODE")
        self.logger.info("=" * 60)
        self.logger.info("Press Ctrl+C to stop monitoring\n")

        try:
            # Start watching the root folder recursively
            watch_path = self.settings.get_scan_root()
            self.observer.schedule(self.event_handler, watch_path, recursive=True)
            self.observer.start()

            self.logger.info(f"👀 Watching for filesystem events in: {watch_path}")

            # Keep the observer running
            while self.running:
                time.sleep(1)

        except Exception as e:
            self.logger.error(f"Error starting filesystem observer: {e}")
            self.running = False
        finally:
            # Stop the observer
            self.observer.stop()
            self.observer.join()

    def run(self) -> int:
        """
        Run the complete pipeline orchestration.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Step 1: Initial full sync
            if not self.run_initial_sync():
                return 1

            # Step 2: Event-driven continuous monitoring
            self.run_monitoring_loop()

            self.logger.info("\n" + "=" * 60)
            self.logger.info("👋 Graceful shutdown completed")
            self.logger.info("=" * 60)
            return 0

        except KeyboardInterrupt:
            return 0
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            return 1
