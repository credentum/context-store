#!/usr/bin/env python3
"""
Tests for config_validator module
"""

import os
import sys
import tempfile
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import yaml  # noqa: E402
from click.testing import CliRunner  # noqa: E402

# Local imports
from src.validators.config_validator import ConfigValidator, main  # noqa: E402


class TestConfigValidator:
    """Tests for ConfigValidator class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.validator = ConfigValidator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up temp files"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_config_file(self, filename, content):
        """Helper to create config files"""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w") as f:
            if isinstance(content, dict):
                yaml.dump(content, f)
            else:
                f.write(content)
        return filepath

    def test_valid_main_config(self) -> None:
        """Test with valid main configuration"""
        config = {
            "system": {"name": "test-system"},
            "qdrant": {"host": "localhost", "port": 6333, "ssl": True},
            "neo4j": {"host": "localhost", "port": 7687, "ssl": True},
            "storage": {"root": "./context"},
            "agents": {"cleanup": {"enabled": True}},
            "redis": {"ssl": True},  # Add Redis with SSL to avoid warning
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        assert self.validator.validate_main_config(config_path) is True
        assert len(self.validator.errors) == 0
        assert len(self.validator.warnings) == 0

    def test_missing_config_file(self) -> None:
        """Test with missing configuration file"""
        assert self.validator.validate_main_config("nonexistent.yaml") is False
        assert len(self.validator.errors) == 1
        assert "not found" in self.validator.errors[0]

    def test_invalid_yaml(self) -> None:
        """Test with invalid YAML syntax"""
        config_path = self.create_config_file(".ctxrc.yaml", "invalid: yaml: content:")
        assert self.validator.validate_main_config(config_path) is False
        assert len(self.validator.errors) == 1
        assert "Invalid YAML" in self.validator.errors[0]

    def test_missing_required_sections(self) -> None:
        """Test with missing required sections"""
        config = {
            "system": {"name": "test"},
            # Missing: qdrant, neo4j, storage, agents
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        assert self.validator.validate_main_config(config_path) is False
        assert len(self.validator.errors) == 4
        for section in ["qdrant", "neo4j", "storage", "agents"]:
            assert any(
                section in error for error in self.validator.errors
            )

    def test_invalid_qdrant_port(self) -> None:
        """Test with invalid Qdrant port"""
        # Non-integer port
        config = {
            "system": {"name": "test"},
            "qdrant": {"port": "6333"},  # String instead of int
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        # Create a new validator instance for each test
        validator = ConfigValidator()
        assert validator.validate_main_config(config_path) is False
        assert any(
            "qdrant.port must be an integer" in e for e in validator.errors
        )

        # Port out of range
        # Out of range integer
        config["qdrant"]["port"] = 99999  # type: ignore[assignment]
        config_path = self.create_config_file(".ctxrc2.yaml", config)

        validator2 = ConfigValidator()
        assert validator2.validate_main_config(config_path) is False
        assert any(
            "qdrant.port must be between" in e for e in validator2.errors
        )

    def test_invalid_neo4j_port(self) -> None:
        """Test with invalid Neo4j port"""
        # Non-integer port
        config = {
            "system": {"name": "test"},
            "qdrant": {},
            "neo4j": {"port": "7687"},  # String instead of int
            "storage": {},
            "agents": {},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        validator = ConfigValidator()
        assert validator.validate_main_config(config_path) is False
        assert any(
            "neo4j.port must be an integer" in e for e in validator.errors
        )

        # Port out of range
        # Out of range integer (port 0 is invalid)
        config["neo4j"]["port"] = 0  # type: ignore[assignment]
        config_path = self.create_config_file(".ctxrc2.yaml", config)

        validator2 = ConfigValidator()
        assert validator2.validate_main_config(config_path) is False
        assert any(
            "neo4j.port must be between" in e for e in validator2.errors
        )

    def test_redis_configuration(self) -> None:
        """Test Redis configuration validation"""
        # Invalid port type
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "redis": {"port": "invalid", "database": 0},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        validator = ConfigValidator()
        assert validator.validate_main_config(config_path) is False
        assert any(
            "redis.port must be an integer" in e for e in validator.errors
        )

        # Port out of range
        config["redis"]["port"] = 70000
        config_path = self.create_config_file(".ctxrc2.yaml", config)

        validator2 = ConfigValidator()
        assert validator2.validate_main_config(config_path) is False
        assert any(
            "redis.port must be between" in e for e in validator2.errors
        )

        # Invalid database
        config["redis"] = {"port": 6379, "database": -1}
        config_path = self.create_config_file(".ctxrc3.yaml", config)

        validator3 = ConfigValidator()
        assert validator3.validate_main_config(config_path) is False
        assert any(
            "redis.database must be a non-negative" in e for e in validator3.errors
        )

    def test_duckdb_configuration(self) -> None:
        """Test DuckDB configuration validation"""
        # Missing database_path
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "duckdb": {"threads": 4},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_main_config(config_path) is False
        assert any(
            "duckdb.database_path is required" in e for e in self.validator.errors
        )

        # Invalid threads
        invalid_duckdb_config: Any = {
            "database_path": "/tmp/db.duckdb",
            "threads": "0",
        }  # Invalid type for threads (should be int)
        config["duckdb"] = invalid_duckdb_config
        config_path = self.create_config_file(".ctxrc2.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_main_config(config_path) is False
        assert any(
            "duckdb.threads must be a positive" in e for e in self.validator.errors
        )

    def test_ssl_warnings(self) -> None:
        """Test SSL warnings for services"""
        config = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "storage": {},
            "agents": {},
            "redis": {"ssl": False},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)
        self.validator.errors = []
        self.validator.warnings = []

        assert self.validator.validate_main_config(config_path) is True
        assert len(self.validator.warnings) == 3
        assert any("SSL is disabled for Qdrant" in w for w in self.validator.warnings)
        assert any("SSL is disabled for Neo4j" in w for w in self.validator.warnings)
        assert any("SSL is disabled for Redis" in w for w in self.validator.warnings)

    def test_performance_config_missing(self) -> None:
        """Test that missing performance config is okay"""
        assert self.validator.validate_performance_config("nonexistent.yaml") is True

    def test_performance_config_invalid_yaml(self) -> None:
        """Test with invalid YAML in performance config"""
        config_path = self.create_config_file(
            "performance.yaml", "invalid: yaml: content:"
        )
        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "Invalid YAML" in e for e in self.validator.errors
        )

    def test_vector_db_embedding_validation(self) -> None:
        """Test vector DB embedding settings validation"""
        config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 0,  # Invalid
                    "max_retries": -1,  # Invalid
                    "request_timeout": 0,  # Invalid
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "batch_size must be a positive" in e for e in self.validator.errors
        )
        assert any(
            "max_retries must be a non-negative" in e for e in self.validator.errors
        )
        assert any(
            "request_timeout must be a positive" in e for e in self.validator.errors
        )

    def test_vector_db_search_validation(self) -> None:
        """Test vector DB search settings validation"""
        config = {
            "vector_db": {
                "search": {
                    "default_limit": 20,
                    "max_limit": 10,  # max < default is invalid
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "max_limit must be >= default_limit" in e for e in self.validator.errors
        )

    def test_graph_db_connection_pool(self) -> None:
        """Test graph DB connection pool validation"""
        config = {
            "graph_db": {
                "connection_pool": {
                    "min_size": 10,
                    "max_size": 5,  # max < min is invalid
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "max_size must be >= min_size" in e for e in self.validator.errors
        )

    def test_graph_db_query_settings(self) -> None:
        """Test graph DB query settings validation"""
        config = {"graph_db": {"query": {"max_path_length": 0}}}  # Invalid
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "max_path_length must be a positive" in e for e in self.validator.errors
        )

        # Test warning for large path length
        config["graph_db"]["query"]["max_path_length"] = 15
        config_path = self.create_config_file("performance2.yaml", config)
        self.validator.errors = []
        self.validator.warnings = []

        assert self.validator.validate_performance_config(config_path) is True
        assert any(
            "max_path_length > 10 may cause" in w for w in self.validator.warnings
        )

    def test_search_ranking_validation(self) -> None:
        """Test search ranking settings validation"""
        # Test with valid number out of range
        config = {"search": {"ranking": {"temporal_decay_rate": 1.5}}}  # Out of range
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "temporal_decay_rate must be between 0 and 1" in e
            for e in self.validator.errors
        )

        # Test with negative value
        config["search"]["ranking"]["temporal_decay_rate"] = -0.5
        config_path = self.create_config_file("performance2.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "temporal_decay_rate must be between 0 and 1" in e
            for e in self.validator.errors
        )

        # Test with non-numeric value (covers line 163)
        # Test with non-numeric value
        config["search"]["ranking"]["temporal_decay_rate"] = (
            "not-a-number"  # type: ignore
        )
        config_path = self.create_config_file("performance3.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "temporal_decay_rate must be a number" in e for e in self.validator.errors
        )

    def test_type_boosts_validation(self) -> None:
        """Test type boosts validation"""
        config = {
            "search": {
                "ranking": {
                    "type_boosts": {
                        "design": 2.0,  # Valid
                        "decision": -1.0,  # Invalid negative
                        "trace": "high",  # Invalid type
                    }
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "type_boosts.decision must be a non-negative" in e
            for e in self.validator.errors
        )
        assert any(
            "type_boosts.trace must be a non-negative" in e
            for e in self.validator.errors
        )

    def test_resources_validation(self) -> None:
        """Test resources validation"""
        config = {
            "resources": {
                "max_memory_gb": 0.3,  # Too low
                "max_cpu_percent": 150,  # Too high
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "max_memory_gb must be at least 0.5" in e for e in self.validator.errors
        )
        assert any(
            "max_cpu_percent must be between 1 and 100" in e
            for e in self.validator.errors
        )

    def test_kv_store_redis_validation(self) -> None:
        """Test KV store Redis settings validation"""
        config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {"min_size": 10, "max_size": 5},  # Invalid
                    "cache": {"ttl_seconds": 0},  # Invalid
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "redis.connection_pool.max_size must be >= min_size" in e
            for e in self.validator.errors
        )
        assert any(
            "ttl_seconds must be a positive" in e for e in self.validator.errors
        )

    def test_kv_store_duckdb_validation(self) -> None:
        """Test KV store DuckDB settings validation"""
        config = {
            "kv_store": {
                "duckdb": {
                    "batch_insert": {"size": 0},  # Invalid
                    "analytics": {"retention_days": -1},  # Invalid
                }
            }
        }
        config_path = self.create_config_file("performance.yaml", config)
        self.validator.errors = []

        assert self.validator.validate_performance_config(config_path) is False
        assert any(
            "batch_insert.size must be a positive" in e for e in self.validator.errors
        )
        assert any(
            "retention_days must be a positive" in e for e in self.validator.errors
        )

    def test_validate_all(self) -> None:
        """Test validate_all method"""
        # Create valid main config
        main_config: Dict[str, Any] = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        self.create_config_file(".ctxrc.yaml", main_config)

        # Create invalid performance config
        perf_config = {"resources": {"max_memory_gb": 0.1}}
        self.create_config_file("performance.yaml", perf_config)

        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(self.temp_dir)

        try:
            valid, errors, warnings = self.validator.validate_all()
            assert valid is False
            assert len(errors) > 0
        finally:
            os.chdir(original_dir)


class TestCLI:
    """Tests for CLI interface"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up temp files"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_config_file(self, filename, content):
        """Helper to create config files"""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w") as f:
            yaml.dump(content, f)
        return filepath

    def test_cli_valid_config(self) -> None:
        """Test CLI with valid configuration"""
        config = {
            "system": {},
            "qdrant": {"ssl": True},
            "neo4j": {"ssl": True},
            "storage": {},
            "agents": {},
            "redis": {"ssl": True},  # Add Redis with SSL to avoid warning
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        result = self.runner.invoke(main, ["--config", config_path])
        assert result.exit_code == 0
        assert "All configurations are valid" in result.output

    def test_cli_with_errors(self) -> None:
        """Test CLI with configuration errors"""
        config: Dict[str, Any] = {"system": {}}  # Missing required sections
        config_path = self.create_config_file(".ctxrc.yaml", config)

        result = self.runner.invoke(main, ["--config", config_path])
        assert result.exit_code == 1
        assert "Errors:" in result.output
        assert "❌" in result.output

    def test_cli_with_warnings(self) -> None:
        """Test CLI with warnings"""
        config = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        result = self.runner.invoke(main, ["--config", config_path])
        assert result.exit_code == 0
        assert "Warnings:" in result.output
        assert "⚠️" in result.output

    def test_cli_strict_mode(self) -> None:
        """Test CLI in strict mode (warnings as errors)"""
        config = {
            "system": {},
            "qdrant": {"ssl": False},  # Will generate warning
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        config_path = self.create_config_file(".ctxrc.yaml", config)

        result = self.runner.invoke(main, ["--config", config_path, "--strict"])
        assert result.exit_code == 1  # Exits with error due to warnings in strict mode

    def test_cli_custom_performance_config(self) -> None:
        """Test CLI with custom performance config path"""
        main_config: Dict[str, Any] = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        main_path = self.create_config_file(".ctxrc.yaml", main_config)

        perf_config = {"resources": {"max_memory_gb": 4}}
        perf_path = self.create_config_file("custom-perf.yaml", perf_config)

        result = self.runner.invoke(
            main, ["--config", main_path, "--perf-config", perf_path]
        )
        assert result.exit_code == 0
        assert f"Validating {perf_path}" in result.output
