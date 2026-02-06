"""Tracing and metrics for ingestion and query operations."""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error


class MetricsCollector:
    """Collect and aggregate metrics for RAG operations."""

    def __init__(self):
        """Initialize metrics collector."""
        self.operations: List[OperationMetrics] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.errors: Dict[str, int] = defaultdict(int)

    @contextmanager
    def trace(self, operation: str, **metadata):
        """Context manager for tracing an operation.

        Args:
            operation: Name of the operation
            **metadata: Additional metadata

        Yields:
            OperationMetrics instance
        """
        metric = OperationMetrics(
            operation=operation, start_time=time.time(), metadata=metadata
        )

        try:
            yield metric
            metric.finish(success=True)
        except Exception as e:
            metric.finish(success=False, error=str(e))
            raise
        finally:
            self.operations.append(metric)
            self._update_aggregates(metric)

    def _update_aggregates(self, metric: OperationMetrics):
        """Update aggregate statistics."""
        # Update counters
        self.counters[f"{metric.operation}_total"] += 1
        if metric.success:
            self.counters[f"{metric.operation}_success"] += 1
        else:
            self.counters[f"{metric.operation}_failed"] += 1
            self.errors[metric.operation] += 1

        # Update timings
        if metric.duration_ms is not None:
            self.timings[metric.operation].append(metric.duration_ms)

    def increment(self, counter: str, value: int = 1):
        """Increment a counter.

        Args:
            counter: Counter name
            value: Amount to increment
        """
        self.counters[counter] += value

    def record_timing(self, operation: str, duration_ms: float):
        """Record a timing manually.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        self.timings[operation].append(duration_ms)

    def get_summary(self) -> Dict:
        """Get summary statistics.

        Returns:
            Dictionary with aggregated metrics
        """
        summary = {
            "total_operations": len(self.operations),
            "counters": dict(self.counters),
            "errors": dict(self.errors),
            "timings": {},
        }

        # Calculate timing statistics
        for operation, times in self.timings.items():
            if times:
                summary["timings"][operation] = {
                    "count": len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "avg_ms": sum(times) / len(times),
                    "total_ms": sum(times),
                }

        return summary

    def get_recent_operations(self, limit: int = 10) -> List[Dict]:
        """Get recent operations.

        Args:
            limit: Number of operations to return

        Returns:
            List of operation dicts
        """
        recent = self.operations[-limit:]
        return [
            {
                "operation": op.operation,
                "duration_ms": op.duration_ms,
                "success": op.success,
                "error": op.error,
                "metadata": op.metadata,
                "timestamp": datetime.fromtimestamp(op.start_time).isoformat(),
            }
            for op in recent
        ]

    def clear(self):
        """Clear all metrics."""
        self.operations.clear()
        self.counters.clear()
        self.timings.clear()
        self.errors.clear()

    def log_summary(self):
        """Log summary statistics."""
        summary = self.get_summary()
        logger.info("=== Metrics Summary ===")
        logger.info(f"Total operations: {summary['total_operations']}")

        if summary["counters"]:
            logger.info("Counters:")
            for name, value in sorted(summary["counters"].items()):
                logger.info(f"  {name}: {value}")

        if summary["errors"]:
            logger.info("Errors:")
            for name, count in sorted(summary["errors"].items()):
                logger.info(f"  {name}: {count}")

        if summary["timings"]:
            logger.info("Timings:")
            for operation, stats in sorted(summary["timings"].items()):
                logger.info(
                    f"  {operation}: "
                    f"avg={stats['avg_ms']:.2f}ms, "
                    f"min={stats['min_ms']:.2f}ms, "
                    f"max={stats['max_ms']:.2f}ms, "
                    f"count={stats['count']}"
                )


# Global metrics collector instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector.

    Returns:
        Global MetricsCollector instance
    """
    return _metrics


def reset_metrics():
    """Reset global metrics collector."""
    global _metrics
    _metrics = MetricsCollector()
