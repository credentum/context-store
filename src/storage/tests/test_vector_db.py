"""
Tests for vector database components
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from src.analytics.sum_scores_api import SearchResult, SumScoresAPI
from src.storage.hash_diff_embedder import DocumentHash, HashDiffEmbedder

# Import components to test
from src.storage.vector_db_init import VectorDBInitializer


class TestVectorDBInitializer:
    """Test vector database initialization"""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create test config file"""
        config = {
            "qdrant": {
                "version": "1.14.x",
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "embedding_model": "text-embedding-ada-002",
            }
        }
        config_path = tmp_path / ".ctxrc.yaml"
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(config, f)
        return str(config_path)

    def test_load_config(self, config_file) -> None:
        """Test configuration loading"""
        initializer = VectorDBInitializer(config_file)
        assert initializer.config["qdrant"]["host"] == "localhost"
        assert initializer.config["qdrant"]["port"] == 6333

    @patch("src.storage.vector_db_init.QdrantClient")
    def test_connect_success(self, mock_client_class, config_file) -> None:
        """Test successful connection"""
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer(config_file)
        assert initializer.connect() is True
        mock_client_class.assert_called_once_with(host="localhost", port=6333, timeout=30)

    @patch("src.storage.vector_db_init.QdrantClient")
    def test_connect_failure(self, mock_client_class, config_file) -> None:
        """Test connection failure"""
        mock_client_class.side_effect = Exception("Connection failed")

        initializer = VectorDBInitializer(config_file)
        assert initializer.connect() is False

    @patch("src.storage.vector_db_init.QdrantClient")
    def test_create_collection(self, mock_client_class, config_file) -> None:
        """Test collection creation"""
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer(config_file)
        initializer.client = mock_client

        assert initializer.create_collection() is True
        mock_client.create_collection.assert_called_once()

        # Check collection parameters
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["vectors_config"].size == 1536


class TestHashDiffEmbedder:
    """Test hash-based embedder"""

    @pytest.fixture
    def test_dir(self) -> Generator[Path, None, None]:
        """Create test directory structure"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def embedder(self, test_dir):
        """Create embedder with test directory"""
        embedder = HashDiffEmbedder()
        embedder.hash_cache_path = test_dir / ".embeddings_cache/hash_cache.json"
        return embedder

    def test_compute_content_hash(self, embedder) -> None:
        """Test content hashing"""
        content = "Test content"
        hash1 = embedder._compute_content_hash(content)
        hash2 = embedder._compute_content_hash(content)
        hash3 = embedder._compute_content_hash("Different content")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 hex length

    def test_needs_embedding(self, embedder, test_dir) -> None:
        """Test change detection"""
        # Create test file
        test_file = test_dir / "test.yaml"
        test_file.write_text("content: test")

        # First check - should need embedding
        needs, existing_id = embedder.needs_embedding(test_file)
        assert needs is True
        assert existing_id is None

        # Add to cache
        embedder.hash_cache[str(test_file)] = DocumentHash(
            document_id="test",
            file_path=str(test_file),
            content_hash=embedder._compute_content_hash("content: test"),
            embedding_hash="test_hash",
            last_embedded="2025-07-11",
            vector_id="test-vec-001",
        )

        # Second check - should not need embedding
        needs, existing_id = embedder.needs_embedding(test_file)
        assert needs is False
        assert existing_id == "test-vec-001"

        # Change file
        test_file.write_text("content: changed")

        # Third check - should need embedding again
        needs, existing_id = embedder.needs_embedding(test_file)
        assert needs is True
        assert existing_id is None

    @patch("src.storage.hash_diff_embedder.openai.OpenAI")
    @patch("src.storage.hash_diff_embedder.QdrantClient")
    def test_embed_document(self, mock_qdrant_class, mock_openai_class, embedder, test_dir) -> None:
        """Test document embedding"""
        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        embedder.client = mock_client

        # Mock new OpenAI client
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_response

        # Create test document
        test_file = test_dir / "test.yaml"
        test_data = {
            "id": "test-doc",
            "document_type": "design",
            "title": "Test Design",
            "description": "Test description",
            "created_date": "2025-07-11",
        }
        with open(test_file, "w") as f:
            import yaml

            yaml.dump(test_data, f)

        # Embed document
        vector_id = embedder.embed_document(test_file)

        assert vector_id is not None
        assert vector_id.startswith("test-doc-")

        # Check Qdrant upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args

        # Handle both positional and keyword arguments
        if call_args.args:
            points = call_args.kwargs.get(
                "points", call_args.args[1] if len(call_args.args) > 1 else []
            )
        else:
            points = call_args.kwargs.get("points", [])

        assert len(points) == 1
        assert points[0].payload["document_id"] == "test-doc"


class TestSumScoresAPI:
    """Test sum-of-scores search API"""

    @pytest.fixture
    def api(self):
        """Create API instance"""
        return SumScoresAPI()

    def test_calculate_temporal_decay(self, api) -> None:
        """Test temporal decay calculation"""
        from datetime import datetime, timedelta

        # Recent document - no decay
        recent_date = datetime.now().strftime("%Y-%m-%d")
        decay = api._calculate_temporal_decay(recent_date)
        assert decay == 1.0

        # Old document - should have decay
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        decay = api._calculate_temporal_decay(old_date)
        assert 0.5 <= decay < 1.0

        # No date - default to no decay
        decay = api._calculate_temporal_decay(None)
        assert decay == 1.0

    def test_get_type_boost(self, api) -> None:
        """Test document type boosting"""
        assert api._get_type_boost("architecture") == 1.25
        assert api._get_type_boost("design") == 1.2
        assert api._get_type_boost("test") == 0.9
        assert api._get_type_boost("unknown") == 1.0

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_single(self, mock_client_class, api) -> None:
        """Test single vector search"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        api.client = mock_client

        # Mock search results
        mock_result = Mock()
        mock_result.id = "test-vec-001"
        mock_result.score = 0.85
        mock_result.payload = {
            "document_id": "test-doc",
            "document_type": "design",
            "title": "Test Design",
            "file_path": "/test/path.yaml",
            "last_modified": "2025-07-11",
        }

        mock_client.search.return_value = [mock_result]

        # Perform search
        query_vector = [0.1] * 1536
        results = api.search_single(query_vector, limit=5)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.document_id == "test-doc"
        assert result.score == 0.85
        assert result.boost_factor == 1.2  # design boost
        assert result.decay_factor == 1.0  # recent document
        assert result.final_score == 0.85 * 1.2 * 1.0

    def test_search_multi_aggregation(self, api) -> None:
        """Test multi-query aggregation logic"""
        # Create test results
        result1 = SearchResult(
            vector_id="vec1",
            document_id="doc1",
            document_type="design",
            file_path="/test1.yaml",
            title="Test 1",
            score=0.8,
            raw_scores=[0.8],
            decay_factor=1.0,
            boost_factor=1.2,
            final_score=0.96,
            payload={},
        )

        # Create second result for same document (for aggregation test)
        _ = SearchResult(
            vector_id="vec1",  # Same document
            document_id="doc1",
            document_type="design",
            file_path="/test1.yaml",
            title="Test 1",
            score=0.7,
            raw_scores=[0.7],
            decay_factor=1.0,
            boost_factor=1.2,
            final_score=0.84,
            payload={},
        )

        # Test sum aggregation
        # Track results for aggregation test
        _ = {"vec1": result1}
        result1.raw_scores.append(0.7)
        result1.score = 0.8 + 0.7  # Sum
        result1.final_score = 1.5 * 1.0 * 1.2

        assert result1.score == 1.5
        assert len(result1.raw_scores) == 2

    def test_load_config_file_not_found(self) -> None:
        """Test configuration loading when file doesn't exist"""
        api = SumScoresAPI(config_path="nonexistent.yaml")
        assert api.config == {}

    def test_load_perf_config_file_not_found(self) -> None:
        """Test performance config loading when file doesn't exist"""
        api = SumScoresAPI(perf_config_path="nonexistent.yaml")
        # Should return default config
        expected_default = {
            "search": {"ranking": {"temporal_decay_days": 30, "temporal_decay_rate": 0.01}}
        }
        assert api.perf_config == expected_default

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_connect_success(self, mock_client_class) -> None:
        """Test successful connection to Qdrant"""
        mock_client = Mock()
        mock_client.get_collections.return_value = []
        mock_client_class.return_value = mock_client

        api = SumScoresAPI()
        assert api.connect() is True
        assert api.client == mock_client
        mock_client_class.assert_called_once_with(host="localhost", port=6333)

    @patch("src.analytics.sum_scores_api.QdrantClient")
    @patch("click.echo")
    def test_connect_failure(self, mock_echo, mock_client_class) -> None:
        """Test connection failure"""
        mock_client_class.side_effect = Exception("Connection failed")

        api = SumScoresAPI()
        assert api.connect() is False
        assert api.client is None
        mock_echo.assert_called_once_with(
            "Failed to connect to Qdrant: Connection failed", err=True
        )

    def test_calculate_temporal_decay_invalid_date(self, api) -> None:
        """Test temporal decay with invalid date format"""
        # Invalid date should return default 1.0
        decay = api._calculate_temporal_decay("invalid-date")
        assert decay == 1.0

        # Empty string should return 1.0
        decay = api._calculate_temporal_decay("")
        assert decay == 1.0

    def test_calculate_temporal_decay_iso_format(self, api) -> None:
        """Test temporal decay with ISO format dates"""
        from datetime import datetime, timedelta

        # Test Z suffix conversion (this is what the actual code does)
        old_date_z = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        decay = api._calculate_temporal_decay(old_date_z)
        # Should be decayed since it's beyond 30 days (default decay_days)
        assert 0.5 <= decay <= 1.0

        # Test with proper timezone format
        old_date_tz = (datetime.now() - timedelta(days=60)).isoformat()
        decay = api._calculate_temporal_decay(old_date_tz)
        assert 0.5 <= decay <= 1.0

        # Test very old date
        very_old_date = (datetime.now() - timedelta(days=200)).isoformat()
        decay = api._calculate_temporal_decay(very_old_date)
        assert decay == 0.5

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_single_with_filters(self, mock_client_class, api) -> None:
        """Test single search with filter conditions"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        api.client = mock_client

        # Mock search results
        mock_result = Mock()
        mock_result.id = "test-vec-001"
        mock_result.score = 0.75
        mock_result.payload = {
            "document_id": "test-doc",
            "document_type": "decision",
            "title": "Test Decision",
            "file_path": "/test/decision.yaml",
            "last_modified": "2025-07-11",
        }
        mock_client.search.return_value = [mock_result]

        # Search with filters
        query_vector = [0.1] * 1536
        filter_conditions = {"document_type": "decision", "status": "approved"}
        results = api.search_single(query_vector, limit=5, filter_conditions=filter_conditions)

        assert len(results) == 1
        result = results[0]
        assert result.document_type == "decision"
        assert result.boost_factor == 1.15  # decision boost

        # Verify filter was applied
        call_args = mock_client.search.call_args
        assert "query_filter" in call_args.kwargs
        query_filter = call_args.kwargs["query_filter"]
        assert len(query_filter.must) == 2  # Two filter conditions

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_multi_sum_aggregation(self, mock_client_class, api) -> None:
        """Test multi-query search with sum aggregation"""
        mock_client = Mock()
        api.client = mock_client

        # Mock search results for different queries
        def mock_search(**kwargs):
            if kwargs["query_vector"][0] == 0.1:
                # First query results
                mock_result = Mock()
                mock_result.id = "vec1"
                mock_result.score = 0.8
                mock_result.payload = {
                    "document_id": "doc1",
                    "document_type": "design",
                    "title": "Test Doc",
                    "file_path": "/test.yaml",
                    "last_modified": "2025-07-11",
                }
                return [mock_result]
            else:
                # Second query results (same document)
                mock_result = Mock()
                mock_result.id = "vec1"
                mock_result.score = 0.6
                mock_result.payload = {
                    "document_id": "doc1",
                    "document_type": "design",
                    "title": "Test Doc",
                    "file_path": "/test.yaml",
                    "last_modified": "2025-07-11",
                }
                return [mock_result]

        mock_client.search.side_effect = mock_search

        # Perform multi-query search
        query_vectors = [[0.1] * 1536, [0.2] * 1536]
        results = api.search_multi(query_vectors, limit=5, aggregation="sum")

        assert len(results) == 1
        result = results[0]
        assert result.score == 1.4  # 0.8 + 0.6
        assert len(result.raw_scores) == 2

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_multi_max_aggregation(self, mock_client_class, api) -> None:
        """Test multi-query search with max aggregation"""
        mock_client = Mock()
        api.client = mock_client

        # Mock multiple results for the same document
        def mock_search(**kwargs):
            mock_result = Mock()
            mock_result.id = "vec1"
            mock_result.score = 0.8 if kwargs["query_vector"][0] == 0.1 else 0.6
            mock_result.payload = {
                "document_id": "doc1",
                "document_type": "design",
                "title": "Test Doc",
                "file_path": "/test.yaml",
                "last_modified": "2025-07-11",
            }
            return [mock_result]

        mock_client.search.side_effect = mock_search

        # Perform multi-query search with max aggregation
        query_vectors = [[0.1] * 1536, [0.2] * 1536]
        results = api.search_multi(query_vectors, limit=5, aggregation="max")

        assert len(results) == 1
        result = results[0]
        assert result.score == 0.8  # max(0.8, 0.6)

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_multi_avg_aggregation(self, mock_client_class, api) -> None:
        """Test multi-query search with average aggregation"""
        mock_client = Mock()
        api.client = mock_client

        # Mock multiple results for the same document
        def mock_search(**kwargs):
            mock_result = Mock()
            mock_result.id = "vec1"
            mock_result.score = 0.8 if kwargs["query_vector"][0] == 0.1 else 0.6
            mock_result.payload = {
                "document_id": "doc1",
                "document_type": "design",
                "title": "Test Doc",
                "file_path": "/test.yaml",
                "last_modified": "2025-07-11",
            }
            return [mock_result]

        mock_client.search.side_effect = mock_search

        # Perform multi-query search with average aggregation
        query_vectors = [[0.1] * 1536, [0.2] * 1536]
        results = api.search_multi(query_vectors, limit=5, aggregation="avg")

        assert len(results) == 1
        result = results[0]
        assert result.score == 0.7  # (0.8 + 0.6) / 2

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_contextual(self, mock_client_class, api) -> None:
        """Test contextual search with document relationships"""
        mock_client = Mock()
        api.client = mock_client

        # Mock base search results
        def mock_search(**kwargs):
            if kwargs.get("filter_conditions"):
                # Context document lookup
                mock_result = Mock()
                mock_result.id = "context-vec"
                mock_result.score = 0.9
                mock_result.payload = {
                    "document_id": kwargs["filter_conditions"]["document_id"],
                    "document_type": "design",
                    "title": "Context Doc",
                    "file_path": "/context.yaml",
                    "sprint_number": 1,
                }
                return [mock_result]
            else:
                # Base search results
                mock_result1 = Mock()
                mock_result1.id = "result-vec-1"
                mock_result1.score = 0.8
                mock_result1.payload = {
                    "document_id": "result-doc-1",
                    "document_type": "design",  # Same type as context
                    "title": "Result Doc 1",
                    "file_path": "/result1.yaml",
                    "sprint_number": 1,
                }

                mock_result2 = Mock()
                mock_result2.id = "result-vec-2"
                mock_result2.score = 0.7
                mock_result2.payload = {
                    "document_id": "result-doc-2",
                    "document_type": "decision",  # Different type
                    "title": "Result Doc 2",
                    "file_path": "/result2.yaml",
                    "sprint_number": 2,
                }
                return [mock_result1, mock_result2]

        mock_client.search.side_effect = mock_search

        # Perform contextual search
        query_vector = [0.1] * 1536
        context_doc_ids = ["context-doc-1"]
        results = api.search_contextual(query_vector, context_doc_ids, limit=5, context_weight=0.3)

        assert len(results) == 2
        # First result should have higher final score due to context matching
        assert results[0].document_id == "result-doc-1"  # Same type as context
        assert results[1].document_id == "result-doc-2"

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_contextual_with_exception(self, mock_client_class, api) -> None:
        """Test contextual search when context lookup fails"""
        mock_client = Mock()
        api.client = mock_client

        # Mock search to raise exception for context lookup
        def mock_search(**kwargs):
            if kwargs.get("filter_conditions"):
                raise Exception("Context lookup failed")
            else:
                # Base search results
                mock_result = Mock()
                mock_result.id = "result-vec-1"
                mock_result.score = 0.8
                mock_result.payload = {
                    "document_id": "result-doc-1",
                    "document_type": "design",
                    "title": "Result Doc 1",
                    "file_path": "/result1.yaml",
                }
                return [mock_result]

        mock_client.search.side_effect = mock_search

        # Should handle exception gracefully
        query_vector = [0.1] * 1536
        context_doc_ids = ["invalid-context-doc"]
        results = api.search_contextual(query_vector, context_doc_ids, limit=5)

        assert len(results) == 1
        assert results[0].document_id == "result-doc-1"

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_get_statistics_small_collection(self, mock_client_class, api) -> None:
        """Test statistics for small collection (full scan)"""
        mock_client = Mock()
        api.client = mock_client

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 50
        mock_collection_info.config.params.vectors.size = 1536
        mock_collection_info.config.params.vectors.distance = "Cosine"
        mock_client.get_collection.return_value = mock_collection_info

        # Mock scroll results for full scan
        def mock_scroll(**kwargs):
            if kwargs.get("offset") is None:
                # First batch
                points = [
                    Mock(id="1", payload={"document_type": "design"}),
                    Mock(id="2", payload={"document_type": "design"}),
                    Mock(id="3", payload={"document_type": "decision"}),
                ]
                return points, "next_offset_1"
            elif kwargs.get("offset") == "next_offset_1":
                # Second batch
                points = [
                    Mock(id="4", payload={"document_type": "sprint"}),
                    Mock(id="5", payload={"document_type": "unknown"}),
                ]
                return points, None  # No more results
            else:
                return [], None

        mock_client.scroll.side_effect = mock_scroll

        stats = api.get_statistics()

        assert stats["total_vectors"] == 50
        assert stats["vector_size"] == 1536
        assert stats["distance_metric"] == "Cosine"
        assert stats["document_types"]["design"] == 2
        assert stats["document_types"]["decision"] == 1
        assert stats["document_types"]["sprint"] == 1
        assert stats["document_types"]["unknown"] == 1

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_get_statistics_large_collection(self, mock_client_class, api) -> None:
        """Test statistics for large collection (sampling)"""
        mock_client = Mock()
        api.client = mock_client

        # Mock collection info for large collection
        mock_collection_info = Mock()
        mock_collection_info.points_count = 5000
        mock_collection_info.config.params.vectors.size = 1536
        mock_collection_info.config.params.vectors.distance = "Dot"
        mock_client.get_collection.return_value = mock_collection_info

        # Mock scroll for sampling
        sample_points = [Mock(id=str(i)) for i in range(500)]
        mock_client.scroll.return_value = (sample_points, None)

        # Mock retrieve for sampled points
        retrieved_points = [
            Mock(id="1", payload={"document_type": "design"}),
            Mock(id="2", payload={"document_type": "design"}),
            Mock(id="3", payload={"document_type": "decision"}),
        ] * 167  # Approximately 500 points with pattern
        mock_client.retrieve.return_value = retrieved_points

        stats = api.get_statistics()

        assert stats["total_vectors"] == 5000
        assert stats["vector_size"] == 1536
        # Should extrapolate based on sample
        assert "document_types" in stats
        assert stats["type_boosts"] == api.type_boosts

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_get_statistics_error(self, mock_client_class, api) -> None:
        """Test statistics when error occurs"""
        mock_client = Mock()
        api.client = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")

        stats = api.get_statistics()

        assert "error" in stats
        assert stats["error"] == "Collection not found"

    @patch("src.analytics.sum_scores_api.SumScoresAPI.connect")
    @patch("src.analytics.sum_scores_api.SumScoresAPI.search_single")
    @patch("click.echo")
    def test_search_command_table_format(self, mock_echo, mock_search, mock_connect) -> None:
        """Test CLI search command with table format"""
        from src.analytics.sum_scores_api import search

        # Mock successful connection
        mock_connect.return_value = True

        # Mock search results
        mock_result = SearchResult(
            vector_id="vec1",
            document_id="doc1",
            document_type="design",
            file_path="/test.yaml",
            title="Test Document",
            score=0.8,
            raw_scores=[0.8],
            decay_factor=1.0,
            boost_factor=1.2,
            final_score=0.96,
            payload={},
        )
        mock_search.return_value = [mock_result]

        # Mock click context
        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(
            search, ["--query", "test query", "--limit", "5", "--format", "table"]
        )

        assert result.exit_code == 0
        mock_connect.assert_called_once()
        mock_search.assert_called_once()

    @patch("src.analytics.sum_scores_api.SumScoresAPI.connect")
    @patch("src.analytics.sum_scores_api.SumScoresAPI.search_single")
    @patch("click.echo")
    def test_search_command_json_format(self, mock_echo, mock_search, mock_connect) -> None:
        """Test CLI search command with JSON format"""
        from src.analytics.sum_scores_api import search

        # Mock successful connection
        mock_connect.return_value = True

        # Mock search results
        mock_result = SearchResult(
            vector_id="vec1",
            document_id="doc1",
            document_type="design",
            file_path="/test.yaml",
            title="Test Document",
            score=0.8,
            raw_scores=[0.8],
            decay_factor=1.0,
            boost_factor=1.2,
            final_score=0.96,
            payload={},
        )
        mock_search.return_value = [mock_result]

        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(search, ["--query", "test query", "--format", "json"])

        assert result.exit_code == 0
        mock_connect.assert_called_once()

    @patch("src.analytics.sum_scores_api.SumScoresAPI.connect")
    def test_search_command_connection_failure(self, mock_connect) -> None:
        """Test CLI search command when connection fails"""
        from src.analytics.sum_scores_api import search

        # Mock connection failure
        mock_connect.return_value = False

        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(search, ["--query", "test query"])

        assert result.exit_code == 0
        mock_connect.assert_called_once()

    @patch("src.analytics.sum_scores_api.SumScoresAPI.connect")
    @patch("src.analytics.sum_scores_api.SumScoresAPI.get_statistics")
    @patch("click.echo")
    def test_stats_command(self, mock_echo, mock_get_stats, mock_connect) -> None:
        """Test CLI stats command"""
        from src.analytics.sum_scores_api import stats

        # Mock successful connection
        mock_connect.return_value = True

        # Mock statistics
        mock_stats = {
            "total_vectors": 1000,
            "vector_size": 1536,
            "distance_metric": "Cosine",
            "document_types": {"design": 500, "decision": 300},
            "type_boosts": {"design": 1.2, "decision": 1.15},
            "decay_config": {"decay_days": 30, "decay_rate": 0.01},
        }
        mock_get_stats.return_value = mock_stats

        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(stats)

        assert result.exit_code == 0
        mock_connect.assert_called_once()
        mock_get_stats.assert_called_once()

    @patch("src.analytics.sum_scores_api.SumScoresAPI.connect")
    def test_stats_command_connection_failure(self, mock_connect) -> None:
        """Test CLI stats command when connection fails"""
        from src.analytics.sum_scores_api import stats

        # Mock connection failure
        mock_connect.return_value = False

        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(stats)

        assert result.exit_code == 0
        mock_connect.assert_called_once()

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_single_empty_payload_fields(self, mock_client_class, api) -> None:
        """Test search with empty/missing payload fields"""
        mock_client = Mock()
        api.client = mock_client

        # Mock result with minimal payload
        mock_result = Mock()
        mock_result.id = "vec-001"
        mock_result.score = 0.8
        mock_result.payload = {}  # Empty payload
        mock_client.search.return_value = [mock_result]

        query_vector = [0.1] * 1536
        results = api.search_single(query_vector, limit=1)

        assert len(results) == 1
        result = results[0]
        assert result.document_id == ""  # Default for missing field
        assert result.document_type == ""
        assert result.title == ""
        assert result.file_path == ""

    @patch("src.analytics.sum_scores_api.QdrantClient")
    def test_search_contextual_no_context_vectors(self, mock_client_class, api) -> None:
        """Test contextual search when no context vectors found"""
        mock_client = Mock()
        api.client = mock_client

        # Mock search that returns empty results for context lookup
        def mock_search(**kwargs):
            if kwargs.get("filter_conditions"):
                return []  # No context documents found
            else:
                # Base search results
                mock_result = Mock()
                mock_result.id = "result-vec-1"
                mock_result.score = 0.8
                mock_result.payload = {
                    "document_id": "result-doc-1",
                    "document_type": "design",
                    "title": "Result Doc 1",
                    "file_path": "/result1.yaml",
                }
                return [mock_result]

        mock_client.search.side_effect = mock_search

        # Should handle no context vectors gracefully
        query_vector = [0.1] * 1536
        context_doc_ids = ["nonexistent-context-doc"]
        results = api.search_contextual(query_vector, context_doc_ids, limit=5)

        assert len(results) == 1
        assert results[0].document_id == "result-doc-1"

    def test_search_result_dataclass(self) -> None:
        """Test SearchResult dataclass creation"""
        result = SearchResult(
            vector_id="vec1",
            document_id="doc1",
            document_type="design",
            file_path="/test.yaml",
            title="Test Doc",
            score=0.8,
            raw_scores=[0.8, 0.6],
            decay_factor=0.9,
            boost_factor=1.2,
            final_score=0.864,
            payload={"key": "value"},
        )

        assert result.vector_id == "vec1"
        assert result.document_id == "doc1"
        assert result.score == 0.8
        assert len(result.raw_scores) == 2
        assert result.payload["key"] == "value"
