#!/usr/bin/env python3
"""
Comprehensive tests for src/storage/vector_db_init.py
Storage critical domain tests to boost coverage above 78.5% threshold
"""

import tempfile
from unittest.mock import Mock, call, patch

import pytest
import yaml

from src.storage.vector_db_init import VectorDBInitializer


class TestVectorDBInitializer:
    """Test VectorDBInitializer class for Qdrant setup"""

    def test_init_with_default_config(self):
        """Test initialization with default config path"""
        with patch.object(VectorDBInitializer, "_load_config") as mock_load:
            mock_load.return_value = {"qdrant": {"host": "localhost"}}
            initializer = VectorDBInitializer()
            mock_load.assert_called_once_with(".ctxrc.yaml")
            assert initializer.config == {"qdrant": {"host": "localhost"}}
            assert initializer.client is None

    def test_init_with_custom_config_path(self):
        """Test initialization with custom config path"""
        with patch.object(VectorDBInitializer, "_load_config") as mock_load:
            mock_load.return_value = {"test": "config"}
            initializer = VectorDBInitializer("/custom/path.yaml")
            mock_load.assert_called_once_with("/custom/path.yaml")
            assert initializer.config == {"test": "config"}

    def test_load_config_success(self):
        """Test successful config loading"""
        config_data = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            initializer = VectorDBInitializer(temp_path)
            assert initializer.config == config_data
        finally:
            import os

            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist"""
        with patch("click.echo") as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                VectorDBInitializer("/nonexistent/path.yaml")

            assert exc_info.value.code == 1
            mock_echo.assert_called_with("Error: /nonexistent/path.yaml not found", err=True)

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with patch("click.echo") as mock_echo:
                with pytest.raises(SystemExit) as exc_info:
                    VectorDBInitializer(temp_path)

                assert exc_info.value.code == 1
                assert mock_echo.call_args[0][0].startswith("Error parsing")
        finally:
            import os

            os.unlink(temp_path)

    def test_load_config_non_dict_content(self):
        """Test config loading when YAML doesn't contain a dictionary"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "dict"], f)
            temp_path = f.name

        try:
            with patch("click.echo") as mock_echo:
                with pytest.raises(SystemExit) as exc_info:
                    VectorDBInitializer(temp_path)

                assert exc_info.value.code == 1
                mock_echo.assert_called_with(
                    f"Error: {temp_path} must contain a dictionary", err=True
                )
        finally:
            import os

            os.unlink(temp_path)

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("click.echo")
    def test_connect_success_no_ssl(self, mock_echo, mock_client_class, mock_get_config):
        """Test successful connection without SSL"""
        # Setup mocks
        mock_get_config.return_value = {
            "host": "localhost",
            "port": 6333,
            "ssl": False,
            "timeout": 5,
        }

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock()
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"host": "localhost"}}

        # Test connection
        result = initializer.connect()

        assert result is True
        assert initializer.client == mock_client
        mock_client_class.assert_called_once_with(host="localhost", port=6333, timeout=5)
        mock_client.get_collections.assert_called_once()
        mock_echo.assert_called_with("✓ Connected to Qdrant at localhost:6333")

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("click.echo")
    def test_connect_success_with_ssl(self, mock_echo, mock_client_class, mock_get_config):
        """Test successful connection with SSL"""
        mock_get_config.return_value = {
            "host": "secure-host",
            "port": 6334,
            "ssl": True,
            "verify_ssl": True,
            "timeout": 10,
        }

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock()
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"host": "secure-host"}}

        result = initializer.connect()

        assert result is True
        mock_client_class.assert_called_once_with(
            host="secure-host", port=6334, https=True, verify=True, timeout=10
        )

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("click.echo")
    def test_connect_ssl_verify_disabled(self, mock_echo, mock_client_class, mock_get_config):
        """Test SSL connection with verification disabled"""
        mock_get_config.return_value = {
            "host": "secure-host",
            "port": 6334,
            "ssl": True,
            "verify_ssl": False,
            "timeout": 10,
        }

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"host": "secure-host"}}

        initializer.connect()

        mock_client_class.assert_called_once_with(
            host="secure-host", port=6334, https=True, verify=False, timeout=10
        )

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("click.echo")
    def test_connect_default_port(self, mock_echo, mock_client_class, mock_get_config):
        """Test connection with default port"""
        mock_get_config.return_value = {"host": "localhost", "ssl": False, "timeout": 5}

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {}}

        initializer.connect()

        mock_client_class.assert_called_once_with(host="localhost", port=6333, timeout=5)

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("click.echo")
    def test_connect_failure(self, mock_echo, mock_client_class, mock_get_config):
        """Test connection failure"""
        mock_get_config.return_value = {
            "host": "localhost",
            "port": 6333,
            "ssl": False,
            "timeout": 5,
        }

        mock_client_class.side_effect = Exception("Connection failed")

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {}}

        result = initializer.connect()

        assert result is False
        assert initializer.client is None
        mock_echo.assert_called_with(
            "✗ Failed to connect to Qdrant at localhost:6333: Connection failed", err=True
        )

    @patch("src.core.utils.get_secure_connection_config")
    @patch("src.storage.vector_db_init.QdrantClient")
    def test_connect_client_none(self, mock_client_class, mock_get_config):
        """Test connection when client creation returns None"""
        mock_get_config.return_value = {"host": "localhost", "ssl": False, "timeout": 5}
        mock_client_class.return_value = None

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {}}

        result = initializer.connect()

        assert result is False

    @patch("click.echo")
    def test_create_collection_no_client(self, mock_echo):
        """Test collection creation when not connected"""
        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = None

        result = initializer.create_collection()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("click.echo")
    def test_create_collection_already_exists_no_force(self, mock_echo):
        """Test collection creation when collection exists and force=False"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.create_collection(force=False)

        assert result is True
        mock_echo.assert_called_with(
            "Collection 'test_collection' already exists. Use --force to recreate."
        )

    @patch("click.echo")
    @patch("time.sleep")
    def test_create_collection_force_recreate(self, mock_sleep, mock_echo):
        """Test collection recreation with force=True"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.create_collection(force=True)

        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()
        mock_sleep.assert_called_once_with(1)

    @patch("click.echo")
    def test_create_collection_new(self, mock_echo):
        """Test creation of new collection"""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "new_collection"}}
        initializer.client = mock_client

        result = initializer.create_collection()

        assert result is True
        mock_client.create_collection.assert_called_once()
        # Verify the collection creation parameters
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "new_collection"

    @patch("click.echo")
    def test_create_collection_default_name(self, mock_echo):
        """Test collection creation with default name"""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {}}  # No collection_name specified
        initializer.client = mock_client

        result = initializer.create_collection()

        assert result is True
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "project_context"

    @patch("click.echo")
    def test_create_collection_exception(self, mock_echo):
        """Test collection creation failure due to exception"""
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Creation failed")

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.create_collection()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to create collection: Creation failed", err=True)

    @patch("click.echo")
    def test_verify_setup_no_client(self, mock_echo):
        """Test setup verification when not connected"""
        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = None

        result = initializer.verify_setup()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("click.echo")
    def test_verify_setup_success_vector_params(self, mock_echo):
        """Test successful setup verification with VectorParams"""
        from qdrant_client.models import Distance, VectorParams

        mock_client = Mock()
        mock_info = Mock()
        mock_info.config.params.vectors = VectorParams(size=1536, distance=Distance.COSINE)
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection", "version": "1.14.0"}}
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is True
        mock_client.get_collection.assert_called_once_with("test_collection")

        # Check that information was printed
        expected_calls = [
            call("\nCollection Info:"),
            call("  Name: test_collection"),
            call("  Vector size: 1536"),
            call(f"  Distance metric: {Distance.COSINE}"),
            call("  Points count: 100"),
            call("\nExpected Qdrant version: 1.14.0"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    @patch("click.echo")
    def test_verify_setup_dict_vectors(self, mock_echo):
        """Test setup verification with dictionary vectors config"""
        from qdrant_client.models import Distance

        mock_client = Mock()
        mock_info = Mock()

        # Create mock vector params
        mock_vector_params = Mock()
        mock_vector_params.size = 768
        mock_vector_params.distance = Distance.DOT

        mock_info.config.params.vectors = {"default": mock_vector_params}
        mock_info.points_count = 50
        mock_client.get_collection.return_value = mock_info

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is True
        expected_calls = [
            call("\nCollection Info:"),
            call("  Name: test_collection"),
            call("  Vector 'default' size: 768"),
            call(f"  Vector 'default' distance: {Distance.DOT}"),
            call("  Points count: 50"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    @patch("click.echo")
    def test_verify_setup_exception(self, mock_echo):
        """Test setup verification failure due to exception"""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Verification failed")

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to verify setup: Verification failed", err=True)

    @patch("click.echo")
    def test_insert_test_point_no_client(self, mock_echo):
        """Test test point insertion when not connected"""
        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = None

        result = initializer.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("click.echo")
    @patch("random.random")
    def test_insert_test_point_success(self, mock_random, mock_echo):
        """Test successful test point insertion and retrieval"""
        # Mock random values for consistent testing
        mock_random.side_effect = [0.1] * 1536  # 1536 values of 0.1

        mock_client = Mock()
        mock_result = Mock()
        mock_result.id = "test-point-001"
        mock_client.search.return_value = [mock_result]

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is True
        mock_client.upsert.assert_called_once()
        mock_client.search.assert_called_once()
        mock_client.delete.assert_called_once()
        mock_echo.assert_called_with("✓ Test point inserted and retrieved successfully")

    @patch("click.echo")
    @patch("random.random")
    def test_insert_test_point_verification_failed(self, mock_random, mock_echo):
        """Test test point insertion when verification fails"""
        mock_random.side_effect = [0.1] * 1536

        mock_client = Mock()
        mock_result = Mock()
        mock_result.id = "wrong-id"  # Different ID
        mock_client.search.return_value = [mock_result]

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("click.echo")
    @patch("random.random")
    def test_insert_test_point_no_results(self, mock_random, mock_echo):
        """Test test point insertion when search returns no results"""
        mock_random.side_effect = [0.1] * 1536

        mock_client = Mock()
        mock_client.search.return_value = []  # No results

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("click.echo")
    def test_insert_test_point_exception(self, mock_echo):
        """Test test point insertion failure due to exception"""
        mock_client = Mock()
        mock_client.upsert.side_effect = Exception("Insert failed")

        initializer = VectorDBInitializer()
        initializer.config = {"qdrant": {"collection_name": "test_collection"}}
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to test point operations: Insert failed", err=True)

    def test_config_fallback_values(self):
        """Test configuration fallback values"""
        # Test with minimal config
        with patch.object(VectorDBInitializer, "_load_config") as mock_load:
            mock_load.return_value = {}
            initializer = VectorDBInitializer()

            # Test default collection name fallback in create_collection
            with patch("click.echo"):
                initializer.client = Mock()
                initializer.client.get_collections.return_value.collections = []
                initializer.create_collection()

                call_args = initializer.client.create_collection.call_args
                assert call_args[1]["collection_name"] == "project_context"
