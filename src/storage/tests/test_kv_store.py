#!/usr/bin/env python3
"""
test_kv_store.py: Tests for the Key-Value store components
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import redis

from src.analytics.context_analytics import ContextAnalytics
from src.storage.context_kv import ContextKV, DuckDBAnalytics, MetricEvent, RedisConnector


class TestRedisConnector:
    """Test Redis connector functionality"""

    @pytest.fixture
    def redis_connector(self):
        """Create Redis connector instance"""
        with patch("src.storage.context_kv.redis.ConnectionPool"):
            connector = RedisConnector(verbose=True)
            connector.redis_client = MagicMock(spec=redis.Redis)
            connector.is_connected = True
            return connector

    def test_connect_success(self) -> None:
        """Test successful Redis connection"""
        with patch("src.storage.context_kv.redis.ConnectionPool"):
            with patch("src.storage.context_kv.redis.Redis") as mock_redis:
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_redis.return_value = mock_client

                connector = RedisConnector()
                assert connector.connect() is True
                assert connector.is_connected is True
                mock_client.ping.assert_called_once()

    def test_connect_failure(self) -> None:
        """Test Redis connection failure"""
        with patch("src.storage.context_kv.redis.ConnectionPool"):
            with patch("src.storage.context_kv.redis.Redis") as mock_redis:
                mock_redis.side_effect = Exception("Connection failed")

                connector = RedisConnector()
                assert connector.connect() is False
                assert connector.is_connected is False

    def test_set_cache(self, redis_connector) -> None:
        """Test setting cache value"""
        key = "test:key"
        value = {"data": "test", "count": 42}
        ttl = 3600

        redis_connector.redis_client.setex.return_value = True

        assert redis_connector.set_cache(key, value, ttl) is True

        # Verify the call
        redis_connector.redis_client.setex.assert_called_once()
        call_args = redis_connector.redis_client.setex.call_args

        assert call_args[0][0] == "cache:test:key"
        assert call_args[0][1] == ttl

        # Check the stored data structure
        stored_data = json.loads(call_args[0][2])
        assert stored_data["key"] == key
        assert stored_data["value"] == value
        assert stored_data["ttl_seconds"] == ttl

    def test_get_cache(self, redis_connector) -> None:
        """Test getting cache value"""
        key = "test:key"
        cached_data = {
            "key": key,
            "value": {"data": "test"},
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
            "hit_count": 2,
            "last_accessed": None,
        }

        redis_connector.redis_client.get.return_value = json.dumps(cached_data)
        redis_connector.redis_client.ttl.return_value = 1800

        result = redis_connector.get_cache(key)

        assert result == {"data": "test"}
        redis_connector.redis_client.get.assert_called_with("cache:test:key")

    def test_delete_cache(self, redis_connector) -> None:
        """Test deleting cache entries"""
        pattern = "test:*"
        matching_keys = ["cache:test:1", "cache:test:2", "cache:test:3"]

        redis_connector.redis_client.scan_iter.return_value = matching_keys
        redis_connector.redis_client.delete.return_value = 3

        result = redis_connector.delete_cache(pattern)

        assert result == 3
        redis_connector.redis_client.scan_iter.assert_called_with(match="cache:test:*")
        redis_connector.redis_client.delete.assert_called_with(*matching_keys)

    def test_session_management(self, redis_connector) -> None:
        """Test session storage and retrieval"""
        session_id = "session123"
        session_data = {"user": "test", "permissions": ["read", "write"]}
        ttl = 3600

        # Test set session
        redis_connector.redis_client.setex.return_value = True
        assert redis_connector.set_session(session_id, session_data, ttl) is True

        # Test get session
        stored_session = {
            "id": session_id,
            "data": session_data,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
        }

        redis_connector.redis_client.get.return_value = json.dumps(stored_session)
        redis_connector.redis_client.ttl.return_value = 1800

        result = redis_connector.get_session(session_id)
        assert result == session_data

    def test_distributed_lock(self, redis_connector) -> None:
        """Test distributed lock acquisition and release"""
        resource = "resource:123"
        timeout = 30

        # Test acquire lock
        redis_connector.redis_client.set.return_value = True
        lock_id = redis_connector.acquire_lock(resource, timeout)

        assert lock_id is not None
        redis_connector.redis_client.set.assert_called_once()
        call_args = redis_connector.redis_client.set.call_args
        assert call_args[0][0] == "lock:resource:123"
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == timeout

        # Test release lock
        redis_connector.redis_client.eval.return_value = 1
        assert redis_connector.release_lock(resource, lock_id) is True

    def test_record_metric(self, redis_connector) -> None:
        """Test metric recording"""
        metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="test.metric",
            value=42.5,
            tags={"env": "test"},
            document_id="doc123",
        )

        redis_connector.redis_client.zadd.return_value = 1
        redis_connector.redis_client.expire.return_value = True

        assert redis_connector.record_metric(metric) is True

        # Verify zadd was called with correct parameters
        redis_connector.redis_client.zadd.assert_called_once()
        call_args = redis_connector.redis_client.zadd.call_args

        metric_key = call_args[0][0]
        assert metric_key.startswith("metric:test.metric:")

        # Verify expiration was set
        redis_connector.redis_client.expire.assert_called_once()


class TestDuckDBAnalytics:
    """Test DuckDB analytics functionality"""

    @pytest.fixture
    def duckdb_analytics(self):
        """Create DuckDB analytics instance"""
        with patch("src.storage.context_kv.duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            analytics = DuckDBAnalytics(verbose=True)
            analytics.conn = mock_conn
            analytics.is_connected = True

            return analytics

    def test_connect_and_initialize(self) -> None:
        """Test DuckDB connection and table initialization"""
        with patch("src.storage.context_kv.duckdb.connect") as mock_connect:
            with patch("src.storage.context_kv.Path.mkdir"):
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn

                analytics = DuckDBAnalytics()
                assert analytics.connect() is True

                # Verify configuration was set
                mock_conn.execute.assert_any_call("SET memory_limit = '2GB'")
                mock_conn.execute.assert_any_call("SET threads = 4")

                # Verify tables were created
                create_calls = [
                    c for c in mock_conn.execute.call_args_list if "CREATE TABLE" in str(c)
                ]
                assert len(create_calls) == 4  # metrics, events, summaries, trends

    def test_insert_metrics(self, duckdb_analytics) -> None:
        """Test batch metric insertion"""
        metrics = [
            MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="test.metric1",
                value=10.5,
                tags={"env": "test"},
                document_id="doc1",
            ),
            MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="test.metric2",
                value=20.5,
                tags={"env": "prod"},
                document_id="doc2",
            ),
        ]

        duckdb_analytics.conn.executemany.return_value = None

        assert duckdb_analytics.insert_metrics(metrics) is True

        # Verify executemany was called
        duckdb_analytics.conn.executemany.assert_called_once()
        call_args = duckdb_analytics.conn.executemany.call_args

        # Check SQL query
        assert "INSERT INTO" in call_args[0][0]
        assert "context_metrics" in call_args[0][0]

        # Check values
        values = call_args[0][1]
        assert len(values) == 2
        assert values[0][1] == "test.metric1"
        assert values[0][2] == 10.5

    def test_query_metrics(self, duckdb_analytics) -> None:
        """Test metric querying"""
        query = "SELECT * FROM context_metrics WHERE metric_name = ?"
        params = ["test.metric"]

        mock_result = [("2024-01-01 00:00:00", "test.metric", 42.5, "doc1", "agent1", "{}")]

        duckdb_analytics.conn.execute.return_value.fetchall.return_value = mock_result
        duckdb_analytics.conn.description = [
            ("timestamp",),
            ("metric_name",),
            ("value",),
            ("document_id",),
            ("agent_id",),
            ("tags",),
        ]

        results = duckdb_analytics.query_metrics(query, params)

        assert len(results) == 1
        assert results[0]["metric_name"] == "test.metric"
        assert results[0]["value"] == 42.5

    def test_aggregate_metrics(self, duckdb_analytics) -> None:
        """Test metric aggregation"""
        metric_name = "test.metric"
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        mock_result = [(35.5, 100, start_time, end_time)]
        duckdb_analytics.conn.execute.return_value.fetchone.return_value = mock_result[0]

        result = duckdb_analytics.aggregate_metrics(metric_name, start_time, end_time, "avg")

        assert result["aggregation"] == "avg"
        assert result["value"] == 35.5
        assert result["count"] == 100

    def test_generate_summary(self, duckdb_analytics) -> None:
        """Test summary generation"""
        from datetime import date

        summary_date = date.today()

        mock_results = [
            ("metric1", 100, 25.5, 10.0, 50.0, 5.5),
            ("metric2", 200, 35.5, 20.0, 60.0, 8.5),
        ]

        duckdb_analytics.conn.execute.return_value.fetchall.return_value = mock_results

        summary = duckdb_analytics.generate_summary(summary_date, "daily")

        assert summary["summary_type"] == "daily"
        assert "metric1" in summary["metrics"]
        assert summary["metrics"]["metric1"]["count"] == 100
        assert summary["metrics"]["metric1"]["avg"] == 25.5


class TestContextAnalytics:
    """Test advanced analytics functionality"""

    @pytest.fixture
    def context_analytics(self):
        """Create context analytics instance"""
        with patch("src.storage.context_kv.duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            analytics = ContextAnalytics(verbose=True)
            analytics.conn = mock_conn
            analytics.is_connected = True

            return analytics

    def test_analyze_document_lifecycle(self, context_analytics) -> None:
        """Test document lifecycle analysis"""
        mock_results = [
            {
                "day": datetime(2024, 1, 1),
                "active_documents": 100,
                "created": 10,
                "updated": 20,
                "archived": 5,
                "accessed": 150,
            },
            {
                "day": datetime(2024, 1, 2),
                "active_documents": 105,
                "created": 15,
                "updated": 25,
                "archived": 10,
                "accessed": 180,
            },
        ]

        context_analytics.query_metrics = MagicMock(return_value=mock_results)

        report = context_analytics.analyze_document_lifecycle(days=30)

        assert report is not None
        assert report.report_type == "document_lifecycle"
        assert "total_active" in report.metrics
        assert "churn_rate" in report.metrics
        # Insights may be empty if metrics are within normal ranges
        assert isinstance(report.insights, list)

    def test_analyze_agent_performance(self, context_analytics) -> None:
        """Test agent performance analysis"""
        mock_results = [
            {
                "agent_id": "agent1",
                "total_actions": 100,
                "successes": 90,
                "failures": 10,
                "avg_duration": 5.5,
                "last_active": datetime.utcnow().isoformat(),
            },
            {
                "agent_id": "agent2",
                "total_actions": 50,
                "successes": 45,
                "failures": 5,
                "avg_duration": 3.2,
                "last_active": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            },
        ]

        context_analytics.query_metrics = MagicMock(return_value=mock_results)

        report = context_analytics.analyze_agent_performance(days=7)

        assert report is not None
        assert report.report_type == "agent_performance"
        assert report.metrics["total_agents"] == 2
        assert report.metrics["total_actions"] == 150
        assert "agent1" in report.metrics["agent_metrics"]
        assert report.metrics["agent_metrics"]["agent1"]["success_rate"] == 0.9

    def test_analyze_system_health(self, context_analytics) -> None:
        """Test system health analysis"""
        mock_metrics = [
            {
                "metric_name": "system.cpu",
                "avg_value": 45.5,
                "min_value": 20.0,
                "max_value": 75.0,
                "count": 1000,
            },
            {
                "metric_name": "system.memory",
                "avg_value": 60.0,
                "min_value": 40.0,
                "max_value": 85.0,
                "count": 1000,
            },
        ]

        mock_errors = [{"error_count": 25, "warning_count": 100}]

        # Mock both queries
        context_analytics.query_metrics = MagicMock()
        context_analytics.query_metrics.side_effect = [mock_metrics, mock_errors]

        report = context_analytics.analyze_system_health()

        assert report is not None
        assert report.report_type == "system_health"
        assert report.metrics["error_count"] == 25
        assert report.metrics["warning_count"] == 100
        assert "health_score" in report.metrics
        assert report.metrics["health_score"] <= 100


class TestContextKV:
    """Test unified KV store interface"""

    @pytest.fixture
    def context_kv(self):
        """Create context KV instance"""
        kv = ContextKV(verbose=True)
        kv.redis = MagicMock(spec=RedisConnector)
        kv.duckdb = MagicMock(spec=DuckDBAnalytics)
        return kv

    def test_connect(self, context_kv) -> None:
        """Test connecting to both stores"""
        context_kv.redis.connect.return_value = True
        context_kv.duckdb.connect.return_value = True

        assert context_kv.connect() is True

        context_kv.redis.connect.assert_called_once()
        context_kv.duckdb.connect.assert_called_once()

    def test_record_event(self, context_kv) -> None:
        """Test event recording to both stores"""
        context_kv.redis.record_metric.return_value = True
        context_kv.duckdb.insert_metrics.return_value = True

        result = context_kv.record_event(
            "test_event", document_id="doc123", agent_id="agent1", data={"action": "create"}
        )

        assert result is True

        # Verify both stores were called
        context_kv.redis.record_metric.assert_called_once()
        context_kv.duckdb.insert_metrics.assert_called_once()

        # Check the metric event
        redis_call = context_kv.redis.record_metric.call_args[0][0]
        assert redis_call.metric_name == "event.test_event"
        assert redis_call.value == 1.0
        assert redis_call.document_id == "doc123"

    def test_get_recent_activity(self, context_kv) -> None:
        """Test getting recent activity summary"""
        mock_metrics = [
            {"metric_name": "event.create", "count": 50, "avg_value": 1.0},
            {"metric_name": "event.update", "count": 100, "avg_value": 1.0},
        ]

        context_kv.duckdb.query_metrics.return_value = mock_metrics

        activity = context_kv.get_recent_activity(hours=24)

        assert activity["period_hours"] == 24
        assert len(activity["metrics"]) == 2
        assert activity["metrics"][0]["count"] == 50


@pytest.mark.integration
class TestKVStoreIntegration:
    """Integration tests for KV store (requires Redis and DuckDB)"""

    @pytest.mark.skipif(
        not all([os.environ.get("REDIS_HOST"), os.environ.get("DUCKDB_PATH")]),
        reason="Integration test requires REDIS_HOST and DUCKDB_PATH environment variables",
    )
    def test_full_workflow(self) -> None:
        """Test complete KV store workflow"""
        kv = ContextKV()

        # Connect
        assert kv.connect() is True

        try:
            # Record some events
            for i in range(10):
                kv.record_event(f"test_event_{i % 3}", document_id=f"doc{i}", data={"index": i})

            # Test cache
            kv.redis.set_cache("test:integration", {"value": "test"}, 60)
            cached = kv.redis.get_cache("test:integration")
            assert cached == {"value": "test"}

            # Test session
            session_data = {"user": "test", "role": "admin"}
            kv.redis.set_session("session:test", session_data, 3600)
            retrieved = kv.redis.get_session("session:test")
            assert retrieved == session_data

            # Test analytics
            activity = kv.get_recent_activity(hours=1)
            assert len(activity["metrics"]) > 0

            # Cleanup
            kv.redis.delete_cache("test:*")

        finally:
            kv.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
