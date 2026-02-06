"""Parallel document processing with bounded concurrency."""

import concurrent.futures
import queue
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """Process documents in parallel with bounded concurrency and backpressure."""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 100,
        ocr_workers: int = 2,
    ):
        """Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker threads for general processing
            max_queue_size: Maximum queue size for backpressure
            ocr_workers: Number of workers dedicated to OCR (CPU heavy)
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.ocr_workers = ocr_workers
        self.progress_callbacks: List[Callable] = []

    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress callback.

        Args:
            callback: Function called with (processed, total, current_file)
        """
        self.progress_callbacks.append(callback)

    def _notify_progress(self, processed: int, total: int, current_file: str) -> None:
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(processed, total, current_file)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def process_files(
        self,
        files: List[Path],
        process_fn: Callable,
        use_ocr: bool = False,
    ) -> List[Tuple[Path, Optional[Dict], Optional[Exception]]]:
        """Process files in parallel.

        Args:
            files: List of file paths to process
            process_fn: Function that takes a Path and returns result dict
            use_ocr: Whether files may require OCR processing

        Returns:
            List of (file_path, result, error) tuples
        """
        results = []
        total = len(files)
        processed = 0

        # Use fewer workers for OCR-heavy tasks
        workers = self.ocr_workers if use_ocr else self.max_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit tasks with bounded queue using a semaphore
            semaphore = threading.Semaphore(self.max_queue_size)
            futures = {}

            for file_path in files:
                # Acquire semaphore before submitting (backpressure)
                semaphore.acquire()

                future = executor.submit(self._process_with_semaphore, 
                                        process_fn, file_path, semaphore)
                futures[future] = file_path

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                processed += 1

                try:
                    result = future.result()
                    results.append((file_path, result, None))
                    self._notify_progress(processed, total, str(file_path))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append((file_path, None, e))
                    self._notify_progress(processed, total, str(file_path))

        return results

    def _process_with_semaphore(
        self,
        process_fn: Callable,
        file_path: Path,
        semaphore: threading.Semaphore,
    ) -> Optional[Dict]:
        """Process file and release semaphore."""
        try:
            return process_fn(file_path)
        finally:
            semaphore.release()

    def process_batches(
        self,
        items: List,
        process_fn: Callable,
        batch_size: int = 32,
    ) -> List[Tuple[List, Optional[Dict], Optional[Exception]]]:
        """Process items in batches in parallel.

        Args:
            items: List of items to process
            process_fn: Function that takes a batch and returns result dict
            batch_size: Size of each batch

        Returns:
            List of (batch, result, error) tuples
        """
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_fn, batch): batch for batch in batches}

            for future in concurrent.futures.as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    results.append((batch, result, None))
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    results.append((batch, None, e))

        return results
