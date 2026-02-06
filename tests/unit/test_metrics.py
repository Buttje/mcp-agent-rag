"""Tests for metrics collector."""

import time

import pytest

from mcp_agent_rag.rag.metrics import MetricsCollector, get_metrics, reset_metrics


def test_metrics_collector_basic():
    """Test basic metrics collection."""
    metrics = MetricsCollector()

    with metrics.trace("test_operation"):
        time.sleep(0.01)

    summary = metrics.get_summary()
    assert summary["total_operations"] == 1
    assert "test_operation_total" in summary["counters"]
    assert "test_operation" in summary["timings"]


def test_metrics_collector_counters():
    """Test counter tracking."""
    metrics = MetricsCollector()

    metrics.increment("chunks_indexed", 10)
    metrics.increment("cache_hits", 5)
    metrics.increment("cache_misses", 3)

    summary = metrics.get_summary()
    assert summary["counters"]["chunks_indexed"] == 10
    assert summary["counters"]["cache_hits"] == 5
    assert summary["counters"]["cache_misses"] == 3


def test_metrics_collector_timings():
    """Test timing statistics."""
    metrics = MetricsCollector()

    metrics.record_timing("operation1", 100.0)
    metrics.record_timing("operation1", 200.0)
    metrics.record_timing("operation1", 150.0)

    summary = metrics.get_summary()
    assert "operation1" in summary["timings"]
    assert summary["timings"]["operation1"]["count"] == 3
    assert summary["timings"]["operation1"]["min_ms"] == 100.0
    assert summary["timings"]["operation1"]["max_ms"] == 200.0
    assert summary["timings"]["operation1"]["avg_ms"] == 150.0


def test_metrics_collector_errors():
    """Test error tracking."""
    metrics = MetricsCollector()

    try:
        with metrics.trace("failing_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass

    summary = metrics.get_summary()
    assert "failing_operation" in summary["errors"]
    assert summary["errors"]["failing_operation"] == 1


def test_metrics_collector_success_failure():
    """Test success/failure tracking."""
    metrics = MetricsCollector()

    # Successful operation
    with metrics.trace("operation"):
        pass

    # Failed operation
    try:
        with metrics.trace("operation"):
            raise ValueError("Error")
    except ValueError:
        pass

    summary = metrics.get_summary()
    assert summary["counters"]["operation_total"] == 2
    assert summary["counters"]["operation_success"] == 1
    assert summary["counters"]["operation_failed"] == 1


def test_metrics_collector_metadata():
    """Test operation metadata."""
    metrics = MetricsCollector()

    with metrics.trace("operation", file="test.txt", size=1024) as op:
        assert op.metadata["file"] == "test.txt"
        assert op.metadata["size"] == 1024


def test_metrics_collector_recent_operations():
    """Test recent operations tracking."""
    metrics = MetricsCollector()

    for i in range(20):
        with metrics.trace(f"op_{i}"):
            pass

    recent = metrics.get_recent_operations(limit=5)
    assert len(recent) == 5
    # Should have most recent operations
    assert recent[-1]["operation"] == "op_19"


def test_metrics_collector_clear():
    """Test clearing metrics."""
    metrics = MetricsCollector()

    with metrics.trace("operation"):
        pass

    metrics.increment("counter", 10)

    summary = metrics.get_summary()
    assert summary["total_operations"] == 1

    metrics.clear()

    summary = metrics.get_summary()
    assert summary["total_operations"] == 0
    assert len(summary["counters"]) == 0


def test_global_metrics():
    """Test global metrics instance."""
    reset_metrics()

    metrics = get_metrics()
    metrics.increment("test_counter", 5)

    # Get again should return same instance
    metrics2 = get_metrics()
    summary = metrics2.get_summary()
    assert summary["counters"]["test_counter"] == 5

    reset_metrics()


def test_metrics_collector_duration():
    """Test duration calculation."""
    metrics = MetricsCollector()

    with metrics.trace("operation") as op:
        time.sleep(0.05)  # 50ms

    assert op.duration_ms is not None
    assert op.duration_ms >= 50.0  # At least 50ms
    assert op.success is True
