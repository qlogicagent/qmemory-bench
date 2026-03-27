"""Tests for noise generation and cross-time-period testing."""

from __future__ import annotations

import pytest

from qmemory_bench.noise_test import generate_noise_sessions


class TestNoiseGeneration:
    def test_generate_noise_sessions_count(self):
        sessions = generate_noise_sessions(count=10, span_days=30)
        assert len(sessions) == 10

    def test_noise_session_structure(self):
        sessions = generate_noise_sessions(count=5, span_days=7)
        for s in sessions:
            assert s["id"].startswith("noise_")
            assert len(s["messages"]) >= 4  # At least 2 turns × 2 msgs
            assert s["metadata"]["type"] == "noise"
            assert s["metadata"]["generated"] is True
            assert "topic" in s["metadata"]
            assert "timestamp" in s["metadata"]

    def test_noise_sessions_sorted_by_time(self):
        sessions = generate_noise_sessions(count=20, span_days=365)
        timestamps = [s["metadata"]["timestamp"] for s in sessions]
        assert timestamps == sorted(timestamps)

    def test_noise_topic_variety(self):
        """50 sessions should cover at least 3 different topics."""
        sessions = generate_noise_sessions(count=50, span_days=365)
        topics = {s["metadata"]["topic"] for s in sessions}
        assert len(topics) >= 3

    def test_noise_messages_are_filled(self):
        """Templates should be filled — no {placeholder} remains."""
        sessions = generate_noise_sessions(count=20, span_days=30)
        for s in sessions:
            for msg in s["messages"]:
                assert "{" not in msg["content"], (
                    f"Unfilled template: {msg['content']}"
                )
