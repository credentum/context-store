#!/usr/bin/env python3
"""
Comprehensive tests for vector_db_init module
"""

from unittest.mock import Mock, mock_open, patch

import yaml
from click.testing import CliRunner
from qdrant_client.models import Distance

from src.storage.vector_db_init import VectorDBInitializer, main


class TestVectorDBInitializer:
    """Test VectorDBInitializer class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_config = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "api_key": None,
                "timeout": 30,
                "collection_name": "project_context",
            },
            "vector_db": {
                "collection": {
                    "name": "test_collection",
                    "vector": {"size": 1536, "distance": "Cosine"},
                }
            },
            "system": {"project_name": "test_project"},
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_config_success(self, mock_yaml_load, mock_file):
        """Test successful config loading"""
        mock_yaml_load.return_value = self.test_config

        initializer = VectorDBInitializer(".ctxrc.yaml")

        assert initializer.config == self.test_config
        mock_file.assert_called_once_with(".ctxrc.yaml", "r")

    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("sys.exit")
    @patch("click.echo")
    def test_load_config_file_not_found(self, mock_echo, mock_exit, mock_file):
        """Test config loading when file doesn't exist"""
        VectorDBInitializer("nonexistent.yaml")

        mock_echo.assert_called_with("Error: nonexistent.yaml not found", err=True)
        mock_exit.assert_called_with(1)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("sys.exit")
    @patch("click.echo")
    def test_load_config_invalid_yaml(self, mock_echo, mock_exit, mock_yaml_load, mock_file):
        """Test config loading with invalid YAML"""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")

        VectorDBInitializer(".ctxrc.yaml")

        assert any("Error parsing" in str(call) for call in mock_echo.call_args_list)
        mock_exit.assert_called_with(1)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("sys.exit")
    @patch("click.echo")
    def test_load_config_not_dict(self, mock_echo, mock_exit, mock_yaml_load, mock_file):
        """Test config loading when YAML is not a dictionary"""
        mock_yaml_load.return_value = "not a dict"

        VectorDBInitializer(".ctxrc.yaml")

        mock_echo.assert_called_with("Error: .ctxrc.yaml must contain a dictionary", err=True)
        mock_exit.assert_called_with(1)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("src.core.utils.get_secure_connection_config")
    def test_connect_with_ssl(self, mock_get_config, mock_client_class, mock_yaml_load, mock_file):
        """Test connection with SSL enabled"""
        self.test_config["qdrant"]["ssl"] = True  # type: ignore[index]
        self.test_config["qdrant"]["api_key"] = "test_key"  # type: ignore[index]
        mock_yaml_load.return_value = self.test_config

        # Mock get_secure_connection_config to return SSL config
        mock_get_config.return_value = {
            "host": "localhost",
            "port": 6333,
            "ssl": True,
            "api_key": "test_key",
            "timeout": 5,
            "verify_ssl": True,
        }

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        result = initializer.connect()

        assert result is True
        # Check that client was created with SSL parameters
        call_args = mock_client_class.call_args
        assert call_args[1]["host"] == "localhost"
        assert call_args[1]["port"] == 6333
        assert call_args[1]["https"] is True
        assert call_args[1]["verify"] is True

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("src.core.utils.get_secure_connection_config")
    def test_connect_client_none(
        self, mock_get_config, mock_client_class, mock_yaml_load, mock_file
    ):
        """Test connection when client creation returns None"""
        mock_yaml_load.return_value = self.test_config
        mock_get_config.return_value = self.test_config["qdrant"]
        mock_client_class.return_value = None

        initializer = VectorDBInitializer()
        result = initializer.connect()

        assert result is False
        assert initializer.client is None

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_create_collection_not_connected(self, mock_yaml_load, mock_file):
        """Test creating collection when not connected"""
        mock_yaml_load.return_value = self.test_config

        initializer = VectorDBInitializer()
        initializer.client = None

        result = initializer.create_collection()

        assert result is False

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_create_collection_already_exists_no_force(self, mock_echo, mock_yaml_load, mock_file):
        """Test creating collection that already exists without force"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Mock collection already exists
        mock_collection = Mock()
        mock_collection.name = "project_context"  # Use name attribute
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.create_collection(force=False)

        assert result is True
        assert any("already exists" in str(call) for call in mock_echo.call_args_list)
        mock_client.create_collection.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_create_collection_already_exists_with_force(self, mock_yaml_load, mock_file):
        """Test creating collection that already exists with force"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Mock collection already exists
        mock_collection = Mock()
        mock_collection.name = "project_context"  # Use name attribute
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.create_collection(force=True)

        assert result is True
        mock_client.delete_collection.assert_called_once_with("project_context")
        mock_client.create_collection.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_create_collection_exception(self, mock_echo, mock_yaml_load, mock_file):
        """Test create collection with exception"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.side_effect = Exception("Creation failed")

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.create_collection()

        assert result is False
        assert any("Failed to create collection" in str(call) for call in mock_echo.call_args_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_verify_setup_success(self, mock_echo, mock_yaml_load, mock_file):
        """Test successful setup verification"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Mock collection info
        mock_info = Mock()
        mock_info.status = "GREEN"
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.config = Mock(params=Mock(vectors=Mock(size=1536, distance="Cosine")))
        mock_client.get_collection.return_value = mock_info

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is True
        assert any("Collection Info" in str(call) for call in mock_echo.call_args_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_verify_setup_not_connected(self, mock_yaml_load, mock_file):
        """Test verify setup when not connected"""
        mock_yaml_load.return_value = self.test_config

        initializer = VectorDBInitializer()
        initializer.client = None

        result = initializer.verify_setup()

        assert result is False

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_verify_setup_exception(self, mock_echo, mock_yaml_load, mock_file):
        """Test verify setup with exception"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Verification failed")

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is False
        assert any("Failed to verify setup" in str(call) for call in mock_echo.call_args_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_insert_test_point_success(self, mock_yaml_load, mock_file):
        """Test successful test point insertion"""
        mock_yaml_load.return_value = self.test_config

        # Mock Qdrant client
        mock_client = Mock()
        mock_search_result = [Mock(id="test-point-001", score=0.99)]
        mock_client.search.return_value = mock_search_result

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is True
        assert mock_client.upsert.called
        assert mock_client.search.called
        assert mock_client.delete.called

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_insert_test_point_not_connected(self, mock_yaml_load, mock_file):
        """Test insert test point when not connected"""
        mock_yaml_load.return_value = self.test_config

        initializer = VectorDBInitializer()
        initializer.client = None

        result = initializer.insert_test_point()

        assert result is False

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_insert_test_point_exception(self, mock_echo, mock_yaml_load, mock_file):
        """Test insert test point with exception"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Make upsert raise exception
        mock_client.upsert.side_effect = Exception("Insert error")

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is False
        assert any(
            "Failed to test point operations" in str(call) for call in mock_echo.call_args_list
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("src.core.utils.get_secure_connection_config")
    def test_connect_no_ssl(self, mock_get_config, mock_client_class, mock_yaml_load, mock_file):
        """Test connection without SSL"""
        mock_yaml_load.return_value = self.test_config

        # Mock get_secure_connection_config to return non-SSL config
        mock_get_config.return_value = {
            "host": "localhost",
            "port": 6333,
            "ssl": False,
            "timeout": 30,
        }

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        initializer = VectorDBInitializer()
        result = initializer.connect()

        assert result is True
        # Check that client was created without SSL parameters
        call_args = mock_client_class.call_args
        assert call_args[1]["host"] == "localhost"
        assert call_args[1]["port"] == 6333
        assert call_args[1]["timeout"] == 30
        assert "https" not in call_args[1]

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("src.storage.vector_db_init.QdrantClient")
    @patch("src.core.utils.get_secure_connection_config")
    @patch("click.echo")
    def test_connect_exception(
        self, mock_echo, mock_get_config, mock_client_class, mock_yaml_load, mock_file
    ):
        """Test connection with exception"""
        mock_yaml_load.return_value = self.test_config
        mock_get_config.return_value = self.test_config["qdrant"]
        mock_client_class.side_effect = Exception("Connection failed")

        initializer = VectorDBInitializer()
        result = initializer.connect()

        assert result is False
        assert any("Failed to connect" in str(call) for call in mock_echo.call_args_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_create_collection_new(self, mock_yaml_load, mock_file):
        """Test creating new collection"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Mock no existing collections
        mock_client.get_collections.return_value = Mock(collections=[])

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.create_collection()

        assert result is True
        mock_client.create_collection.assert_called_once()
        # Verify collection parameters
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "project_context"
        assert call_args[1]["vectors_config"].size == 1536
        assert call_args[1]["vectors_config"].distance == Distance.COSINE

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("click.echo")
    def test_verify_setup_named_vectors(self, mock_echo, mock_yaml_load, mock_file):
        """Test verify setup with named vectors configuration"""
        mock_yaml_load.return_value = self.test_config

        mock_client = Mock()
        # Mock collection info with named vectors
        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = {
            "default": Mock(size=1536, distance="Cosine"),
            "secondary": Mock(size=768, distance="Dot"),
        }
        mock_info.points_count = 50
        mock_client.get_collection.return_value = mock_info

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is True
        # Check that both vector configs were displayed
        echo_calls = [str(call) for call in mock_echo.call_args_list]
        assert any("default" in call for call in echo_calls)
        assert any("secondary" in call for call in echo_calls)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_insert_test_point_search_verification_fail(self, mock_yaml_load, mock_file):
        """Test insert test point when search verification fails"""
        mock_yaml_load.return_value = self.test_config

        # Mock Qdrant client
        mock_client = Mock()
        # Return empty search results or wrong ID
        mock_client.search.return_value = []

        initializer = VectorDBInitializer()
        initializer.client = mock_client

        result = initializer.insert_test_point()

        assert result is False
        assert mock_client.upsert.called
        assert mock_client.search.called
        # Delete should not be called if verification failed
        assert not mock_client.delete.called


class TestCLI:
    """Test CLI functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_success(self, mock_initializer_class):
        """Test successful main execution"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, [])

        assert result.exit_code == 0
        assert "✓ Qdrant initialization complete!" in result.output

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_connection_failure(self, mock_initializer_class):
        """Test main with connection failure"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = False
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, [])

        assert result.exit_code == 1
        assert "Please ensure Qdrant is running" in result.output

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_with_force(self, mock_initializer_class):
        """Test main with force flag"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, ["--force"])

        assert result.exit_code == 0
        mock_initializer.create_collection.assert_called_with(force=True)

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_skip_test(self, mock_initializer_class):
        """Test main with skip-test flag"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, ["--skip-test"])

        assert result.exit_code == 0
        mock_initializer.insert_test_point.assert_not_called()

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_collection_creation_failure(self, mock_initializer_class):
        """Test main with collection creation failure"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = False
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, [])

        assert result.exit_code == 1

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_verification_failure(self, mock_initializer_class):
        """Test main with verification failure"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = False
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, [])

        assert result.exit_code == 1

    @patch("src.storage.vector_db_init.VectorDBInitializer")
    def test_main_test_point_failure(self, mock_initializer_class):
        """Test main with test point insertion failure"""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = False
        mock_initializer_class.return_value = mock_initializer

        result = self.runner.invoke(main, [])

        assert result.exit_code == 1
        assert "✓ Qdrant initialization complete!" not in result.output
