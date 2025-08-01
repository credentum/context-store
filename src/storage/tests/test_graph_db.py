"""
Tests for graph database components
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.integrations.graphrag_integration import GraphRAGIntegration, GraphRAGResult
from src.storage.graph_builder import GraphBuilder

# Import components to test
from src.storage.neo4j_init import Neo4jInitializer


def create_mock_neo4j_driver():
    """Helper to create properly mocked Neo4j driver with session context manager"""
    mock_driver = Mock()
    mock_session = Mock()
    mock_session_cm = Mock()
    mock_session_cm.__enter__ = Mock(return_value=mock_session)
    mock_session_cm.__exit__ = Mock(return_value=None)
    mock_driver.session.return_value = mock_session_cm
    return mock_driver, mock_session


class TestNeo4jInitializer:
    """Test Neo4j initialization"""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create test config file"""
        config = {
            "neo4j": {
                "version": "5.x",
                "host": "localhost",
                "port": 7687,
                "database": "test_graph",
            },
            "system": {"schema_version": "1.0.0", "created_date": "2025-07-11"},
        }
        config_path = tmp_path / ".ctxrc.yaml"
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(config, f)
        return str(config_path)

    def test_load_config(self, config_file):
        """Test configuration loading"""
        initializer = Neo4jInitializer(config_file)
        assert initializer.config["neo4j"]["host"] == "localhost"
        assert initializer.config["neo4j"]["port"] == 7687
        assert initializer.database == "test_graph"

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_connect_success(self, mock_driver, config_file):
        """Test successful connection"""
        # Setup mock with proper context manager
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_driver.return_value = mock_driver_instance

        initializer = Neo4jInitializer(config_file)
        assert initializer.connect(username="neo4j", password="test") is True

        mock_driver.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "test"))
        mock_session.run.assert_called_once_with("RETURN 1 as test")

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_create_constraints(self, mock_driver, config_file):
        """Test constraint creation"""
        # Setup mock with proper context manager
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_constraints() is True

        # Check constraints were created
        expected_constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Design) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Decision) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Sprint) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Agent) REQUIRE n.name IS UNIQUE",
        ]

        calls = mock_session.run.call_args_list
        for expected in expected_constraints:
            assert any(expected in str(call) for call in calls)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_setup_graph_schema(self, mock_driver, config_file):
        """Test graph schema setup"""
        # Setup mock
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.setup_graph_schema() is True

        # Check system node was created
        calls = mock_session.run.call_args_list
        system_node_created = any(
            "MERGE (s:System {id: 'agent-context-system'})" in str(call) for call in calls
        )
        assert system_node_created

    def test_load_config_file_not_found(self):
        """Test configuration loading when file doesn't exist"""
        with patch("sys.exit") as mock_exit:
            with patch("click.echo") as mock_echo:
                with patch.object(Neo4jInitializer, "_load_config") as mock_load_config:
                    # Make _load_config call sys.exit to simulate the actual behavior
                    def side_effect(path):
                        mock_echo(f"Error: {path} not found", err=True)
                        mock_exit(1)
                        return {}  # Never reached but satisfies type checker

                    mock_load_config.side_effect = side_effect

                    try:
                        Neo4jInitializer("nonexistent.yaml")
                    except SystemExit:
                        pass

                    mock_exit.assert_called_once_with(1)
                    mock_echo.assert_called_with("Error: nonexistent.yaml not found", err=True)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test configuration loading with invalid YAML"""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: {")

        with patch("sys.exit") as mock_exit:
            with patch("click.echo") as mock_echo:
                with patch.object(Neo4jInitializer, "_load_config") as mock_load_config:
                    # Make _load_config call sys.exit to simulate YAML error
                    def side_effect(path):
                        mock_echo(f"Error: {path} must contain a dictionary", err=True)
                        mock_exit(1)
                        return {}

                    mock_load_config.side_effect = side_effect

                    try:
                        Neo4jInitializer(str(invalid_config))
                    except SystemExit:
                        pass

                    mock_exit.assert_called_once_with(1)

    def test_load_config_non_dict(self, tmp_path):
        """Test configuration loading when file contains non-dict"""
        config_file = tmp_path / "non_dict.yaml"
        config_file.write_text("- item1\n- item2")

        with patch("sys.exit") as mock_exit:
            with patch("click.echo") as mock_echo:
                with patch.object(Neo4jInitializer, "_load_config") as mock_load_config:
                    # Make _load_config call sys.exit to simulate non-dict error
                    def side_effect(path):
                        mock_echo(f"Error: {path} must contain a dictionary", err=True)
                        mock_exit(1)
                        return {}

                    mock_load_config.side_effect = side_effect

                    try:
                        Neo4jInitializer(str(config_file))
                    except SystemExit:
                        pass

                    mock_exit.assert_called_once_with(1)
                    mock_echo.assert_called_with(
                        f"Error: {config_file} must contain a dictionary", err=True
                    )

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_connect_ssl_enabled(self, mock_driver, tmp_path):
        """Test connection with SSL enabled"""
        # Create config with SSL
        config = {
            "neo4j": {"host": "localhost", "port": 7687, "ssl": True, "database": "test_graph"}
        }
        config_path = tmp_path / ".ctxrc.yaml"
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(config, f)

        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_driver.return_value = mock_driver_instance

        initializer = Neo4jInitializer(str(config_path))
        assert initializer.connect(username="neo4j", password="test") is True

        mock_driver.assert_called_once_with("bolt+s://localhost:7687", auth=("neo4j", "test"))

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    @patch("getpass.getpass")
    def test_connect_prompt_for_password(self, mock_getpass, mock_echo, mock_driver, config_file):
        """Test connection with password prompt"""
        mock_getpass.return_value = "prompted_password"
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_driver.return_value = mock_driver_instance

        initializer = Neo4jInitializer(config_file)
        assert initializer.connect(username="neo4j", password=None) is True

        mock_getpass.assert_called_once_with("Password: ")
        mock_driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "prompted_password")
        )

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_connect_service_unavailable(self, mock_echo, mock_driver, config_file):
        """Test connection failure when Neo4j is unavailable"""
        from neo4j.exceptions import ServiceUnavailable

        mock_driver.side_effect = ServiceUnavailable("Service unavailable")

        initializer = Neo4jInitializer(config_file)
        assert initializer.connect(username="neo4j", password="test") is False

        mock_echo.assert_any_call("✗ Neo4j is not available at bolt://localhost:7687", err=True)
        mock_echo.assert_any_call("Please ensure Neo4j is running:")

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_connect_auth_error(self, mock_echo, mock_driver, config_file):
        """Test connection failure with authentication error"""
        from neo4j.exceptions import AuthError

        mock_driver.side_effect = AuthError("Authentication failed")

        initializer = Neo4jInitializer(config_file)
        assert initializer.connect(username="neo4j", password="test") is False

        mock_echo.assert_any_call("✗ Authentication failed", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_connect_generic_error(self, mock_echo, mock_driver, config_file):
        """Test connection failure with generic error"""
        mock_driver.side_effect = Exception("Connection failed")

        initializer = Neo4jInitializer(config_file)
        assert initializer.connect(username="neo4j", password="test") is False

        mock_echo.assert_any_call("✗ Failed to connect: Connection failed", err=True)

    @patch("click.echo")
    def test_create_constraints_no_driver(self, mock_echo, config_file):
        """Test constraint creation when not connected"""
        initializer = Neo4jInitializer(config_file)
        assert initializer.create_constraints() is False

        mock_echo.assert_called_once_with("✗ Not connected to Neo4j", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_create_constraints_error(self, mock_echo, mock_driver, config_file):
        """Test constraint creation with database error"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_session.run.side_effect = Exception("Database error")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_constraints() is False
        mock_echo.assert_any_call("✗ Failed to create constraints: Database error", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_create_indexes_success(self, mock_driver, config_file):
        """Test index creation success"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_indexes() is True

        # Verify various index types were created
        calls = mock_session.run.call_args_list
        assert len(calls) > 0

        # Check for regular btree index
        btree_index_found = any(
            "CREATE INDEX" in str(call) and "IF NOT EXISTS" in str(call) for call in calls
        )
        assert btree_index_found

        # Check for fulltext index
        fulltext_index_found = any(
            "db.index.fulltext.createNodeIndex" in str(call) for call in calls
        )
        assert fulltext_index_found

    @patch("click.echo")
    def test_create_indexes_no_driver(self, mock_echo, config_file):
        """Test index creation when not connected"""
        initializer = Neo4jInitializer(config_file)
        assert initializer.create_indexes() is False

        mock_echo.assert_called_once_with("✗ Not connected to Neo4j", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_create_indexes_partial_failure(self, mock_echo, mock_driver, config_file):
        """Test index creation with some failures"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        # Make some index creations fail, others succeed
        def side_effect(query):
            if "fulltext" in query:
                raise Exception("Index creation failed")
            return Mock()

        mock_session.run.side_effect = side_effect

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_indexes() is True  # Should continue despite some failures
        mock_echo.assert_any_call("  Warning: Index creation failed")

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_create_indexes_existing_index(self, mock_echo, mock_driver, config_file):
        """Test index creation when index already exists"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_session.run.side_effect = Exception("already exists")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_indexes() is True  # Should ignore "already exists" errors
        # Should not show warning for "already exists" errors
        warning_calls = [call for call in mock_echo.call_args_list if "Warning:" in str(call)]
        assert len(warning_calls) == 0

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_create_indexes_error(self, mock_echo, mock_driver, config_file):
        """Test index creation with database error"""
        mock_driver_instance = Mock()
        # Make the session context manager itself raise an exception
        mock_driver_instance.session.side_effect = Exception("Database error")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.create_indexes() is False
        mock_echo.assert_any_call("✗ Failed to create indexes: Database error", err=True)

    @patch("click.echo")
    def test_setup_graph_schema_no_driver(self, mock_echo, config_file):
        """Test schema setup when not connected"""
        initializer = Neo4jInitializer(config_file)
        assert initializer.setup_graph_schema() is False

        mock_echo.assert_called_once_with("✗ Not connected to Neo4j", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_setup_graph_schema_error(self, mock_echo, mock_driver, config_file):
        """Test schema setup with database error"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_session.run.side_effect = Exception("Database error")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.setup_graph_schema() is False
        mock_echo.assert_any_call("✗ Failed to setup schema: Database error", err=True)

    @patch("click.echo")
    def test_verify_setup_no_driver(self, mock_echo, config_file):
        """Test verification when not connected"""
        initializer = Neo4jInitializer(config_file)
        assert initializer.verify_setup() is False

        mock_echo.assert_called_once_with("✗ Not connected to Neo4j", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_verify_setup_with_apoc(self, mock_echo, mock_driver, config_file):
        """Test verification with APOC available"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        # Mock APOC query results
        label_records = [
            {"label": "Document", "count": 5},
            {"label": "Agent", "count": 4},
            {"label": "System", "count": 1},
        ]

        rel_records = [{"type": "HAS_AGENT", "count": 4}, {"type": "HAS_DOCUMENT_TYPE", "count": 3}]

        mock_session.run.side_effect = [label_records, rel_records]

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.verify_setup() is True

        mock_echo.assert_any_call("\nNode counts by label:")
        mock_echo.assert_any_call("  Document: 5")
        mock_echo.assert_any_call("\nRelationship counts by type:")

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_verify_setup_fallback_query(self, mock_echo, mock_driver, config_file):
        """Test verification falling back to simple queries when APOC fails"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        # First query (APOC) fails, fallback queries succeed
        mock_node_result = Mock()
        mock_node_result.single.return_value = {"total": 10}

        mock_rel_result = Mock()
        mock_rel_result.single.return_value = {"total": 7}

        mock_session.run.side_effect = [
            Exception("APOC not available"),  # First query fails
            mock_node_result,  # Node count
            mock_rel_result,  # Relationship count
        ]

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.verify_setup() is True

        mock_echo.assert_any_call("\nTotal nodes: 10")
        mock_echo.assert_any_call("Total relationships: 7")

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_verify_setup_complete_failure(self, mock_echo, mock_driver, config_file):
        """Test verification when all queries fail"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_session.run.side_effect = Exception("Database error")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.verify_setup() is False
        mock_echo.assert_any_call("✗ Failed to verify setup: Database error", err=True)

    def test_close_with_driver(self, config_file):
        """Test closing connection when driver exists"""
        initializer = Neo4jInitializer(config_file)
        mock_driver = Mock()
        initializer.driver = mock_driver

        initializer.close()
        mock_driver.close.assert_called_once()

    def test_close_without_driver(self, config_file):
        """Test closing connection when no driver"""
        initializer = Neo4jInitializer(config_file)
        # Should not raise any error
        initializer.close()

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_verify_setup_fallback_failure(self, mock_echo, mock_driver, config_file):
        """Test verification when fallback queries also fail"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        # All queries fail
        mock_session.run.side_effect = Exception("Database error")

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.verify_setup() is False
        mock_echo.assert_any_call("✗ Failed to verify setup: Database error", err=True)

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_verify_setup_fallback_no_records(self, mock_echo, mock_driver, config_file):
        """Test verification fallback with None records"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()

        # APOC fails, fallback returns None records
        mock_node_result = Mock()
        mock_node_result.single.return_value = None

        mock_rel_result = Mock()
        mock_rel_result.single.return_value = None

        mock_session.run.side_effect = [
            Exception("APOC not available"),  # First query fails
            mock_node_result,  # Node count - returns None
            mock_rel_result,  # Relationship count - returns None
        ]

        initializer = Neo4jInitializer(config_file)
        initializer.driver = mock_driver_instance

        assert initializer.verify_setup() is True  # Should still work with None records

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    def test_connect_session_test_failure(self, mock_driver, config_file):
        """Test connection test when session.run fails"""
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        mock_session.run.side_effect = Exception("Session test failed")
        mock_driver.return_value = mock_driver_instance

        initializer = Neo4jInitializer(config_file)
        result = initializer.connect(username="neo4j", password="test")

        # Should fail when session test fails
        assert result is False

    @patch("src.storage.neo4j_init.GraphDatabase.driver")
    @patch("click.echo")
    def test_connect_error_sanitization(self, mock_echo, mock_driver, config_file):
        """Test that sensitive information is sanitized in error messages"""
        mock_driver.side_effect = Exception(
            "Database connection failed with password=secret123 and user=admin"
        )

        initializer = Neo4jInitializer(config_file)
        result = initializer.connect(username="admin", password="secret123")

        assert result is False

        # Check that sensitive values were sanitized
        error_calls = [
            call for call in mock_echo.call_args_list if "✗ Failed to connect:" in str(call)
        ]
        assert len(error_calls) > 0
        error_message = str(error_calls[0])
        assert "secret123" not in error_message  # Password should be sanitized
        assert "admin" not in error_message  # Username should be sanitized

    @patch("src.storage.neo4j_init.Neo4jInitializer")
    @patch("click.echo")
    def test_main_success_with_skips(self, mock_echo, mock_initializer_class):
        """Test main function success with skip flags"""
        from src.storage.neo4j_init import main

        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.driver = Mock()
        mock_initializer.database = "test_db"
        mock_initializer.create_constraints.return_value = True
        mock_initializer.create_indexes.return_value = True
        mock_initializer.setup_graph_schema.return_value = True
        mock_initializer_class.return_value = mock_initializer

        # Mock database session for system database queries
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = None  # Database doesn't exist
        mock_session.run.side_effect = [mock_result, Mock()]  # First for check, second for create
        mock_session_cm = Mock()
        mock_session_cm.__enter__ = Mock(return_value=mock_session)
        mock_session_cm.__exit__ = Mock(return_value=None)
        mock_initializer.driver.session.return_value = mock_session_cm

        if main.callback is not None:
            main.callback("neo4j", "test", True, True, True)

        # Should skip all creation steps
        mock_initializer.create_constraints.assert_not_called()
        mock_initializer.create_indexes.assert_not_called()
        mock_initializer.setup_graph_schema.assert_not_called()
        mock_initializer.verify_setup.assert_called_once()


class TestGraphBuilder:
    """Test graph builder component"""

    @pytest.fixture
    def test_dir(self):
        """Create test directory structure"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def builder(self, test_dir):
        """Create graph builder"""
        builder = GraphBuilder()
        builder.processed_cache_path = test_dir / ".graph_cache/processed.json"
        return builder

    def test_compute_doc_hash(self, builder):
        """Test document hash computation"""
        doc1 = {"id": "test", "content": "test content"}
        doc2 = {"id": "test", "content": "test content"}
        doc3 = {"id": "test", "content": "different content"}

        hash1 = builder._compute_doc_hash(doc1)
        hash2 = builder._compute_doc_hash(doc2)
        hash3 = builder._compute_doc_hash(doc3)

        assert hash1 == hash2  # Same content
        assert hash1 != hash3  # Different content

    @patch("src.storage.graph_builder.GraphDatabase.driver")
    def test_create_document_node(self, mock_driver, builder, test_dir):
        """Test document node creation"""
        # Setup mock
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        builder.driver = mock_driver_instance

        # Test data
        data = {
            "id": "test-design",
            "document_type": "design",
            "title": "Test Design",
            "created_date": "2025-07-11",
            "status": "active",
        }

        file_path = test_dir / "test.yaml"

        # Create node
        doc_id = builder._create_document_node(mock_session, data, file_path)

        assert doc_id == "test-design"

        # Check query
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "Document:Design" in call_args[0][0]
        assert call_args[1]["id"] == "test-design"

    def test_extract_references(self, builder):
        """Test reference extraction from content"""
        content = """
        This references [[design-001]] and @sprint-001.
        Also mentions #decision-002 and [[api-spec]].
        """

        references = builder._extract_references(content)

        assert "design-001" in references
        assert "sprint-001" in references
        assert "decision-002" in references
        assert "api-spec" in references
        assert len(references) == 4

    @patch("src.storage.graph_builder.GraphDatabase.driver")
    def test_process_document(self, mock_driver, builder, test_dir):
        """Test document processing"""
        # Setup mock
        mock_driver_instance, mock_session = create_mock_neo4j_driver()
        builder.driver = mock_driver_instance

        # Create test document
        test_file = test_dir / "test.yaml"
        test_data = {
            "id": "test-sprint",
            "document_type": "sprint",
            "title": "Test Sprint",
            "sprint_number": 1,
            "phases": [
                {"phase": 0, "name": "Setup", "status": "completed", "tasks": ["Task 1", "Task 2"]}
            ],
            "team": [{"role": "lead", "agent": "pm_agent"}],
        }

        with open(test_file, "w") as f:
            import yaml

            yaml.dump(test_data, f)

        # Process document
        success = builder.process_document(test_file)

        assert success is True

        # Check multiple queries were made
        assert mock_session.run.call_count > 1

        # Check cache was updated
        assert str(test_file) in builder.processed_docs


class TestGraphRAGIntegration:
    """Test GraphRAG integration"""

    @pytest.fixture
    def graphrag(self):
        """Create GraphRAG instance"""
        return GraphRAGIntegration()

    def test_extract_reasoning_path(self, graphrag):
        """Test reasoning path extraction"""
        neighborhood = {
            "nodes": {
                "doc1": {"document_type": "design"},
                "doc2": {"document_type": "design"},
                "doc3": {"document_type": "sprint"},
            },
            "relationships": [
                {"type": "IMPLEMENTS", "source": "doc1", "target": "doc2"},
                {"type": "IMPLEMENTS", "source": "doc2", "target": "doc3"},
                {"type": "REFERENCES", "source": "doc1", "target": "doc3"},
            ],
            "paths": [
                {"nodes": ["doc1", "doc2"], "distance": 1},
                {"nodes": ["doc1", "doc3"], "distance": 1},
            ],
        }

        reasoning = graphrag._extract_reasoning_path(neighborhood)

        assert len(reasoning) > 0
        assert any("3 related documents" in r for r in reasoning)
        assert any("2 IMPLEMENTS" in r for r in reasoning)
        assert any("Average connection distance" in r for r in reasoning)

    @patch("src.integrations.graphrag_integration.QdrantClient")
    @patch("src.integrations.graphrag_integration.GraphDatabase.driver")
    def test_search(self, mock_neo4j, mock_qdrant, graphrag):
        """Test GraphRAG search"""
        # Setup mocks
        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance

        mock_neo4j_instance, mock_session = create_mock_neo4j_driver()
        mock_neo4j.return_value = mock_neo4j_instance

        graphrag.qdrant_client = mock_qdrant_instance
        graphrag.neo4j_driver = mock_neo4j_instance

        # Mock vector search results
        mock_vector_result = Mock()
        mock_vector_result.id = "vec1"
        mock_vector_result.score = 0.85
        mock_vector_result.payload = {
            "document_id": "doc1",
            "document_type": "design",
            "title": "Test Design",
            "file_path": "/test.yaml",
        }

        mock_qdrant_instance.search.return_value = [mock_vector_result]

        # Mock graph results
        mock_session.run.return_value = []

        # Perform search
        query = "test query"
        query_vector = [0.1] * 1536

        result = graphrag.search(query, query_vector, max_hops=2, top_k=5)

        assert isinstance(result, GraphRAGResult)
        assert result.query == query
        assert len(result.vector_results) == 1
        assert result.vector_results[0]["document_id"] == "doc1"
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_analyze_document_impact(self, graphrag):
        """Test document impact analysis"""
        with patch.object(graphrag, "neo4j_driver") as mock_driver:
            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session

            # Mock direct connections query
            mock_result1 = Mock()
            mock_result1.single.return_value = {
                "direct_count": 5,
                "connections": [
                    {"id": "doc2", "type": "design", "relationship": "IMPLEMENTS"},
                    {"id": "doc3", "type": "sprint", "relationship": "REFERENCES"},
                ],
            }

            # Mock reachability query
            mock_result2 = Mock()
            mock_result2.single.return_value = {"total": 15}

            # Mock dependency chain query
            mock_session.run.side_effect = [mock_result1, mock_result2, []]

            impact = graphrag.analyze_document_impact("doc1")

            assert impact["document_id"] == "doc1"
            assert impact["direct_connections"] == 5
            assert impact["total_reachable"] == 15
            assert impact["central_score"] == 5 / 15
            assert len(impact["impacted_documents"]) == 2
