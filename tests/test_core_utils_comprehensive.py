#!/usr/bin/env python3
"""
Comprehensive tests for src/core/utils.py
Targeted to boost critical domain coverage above 78.5% threshold
"""

import os
import sys
import warnings
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils import (  # noqa: E402
    get_environment,
    get_secure_connection_config,
    sanitize_error_message,
)


class TestSanitizeErrorMessage:
    """Test error message sanitization for security compliance"""

    def test_empty_message(self):
        """Test handling of empty/None messages"""
        assert sanitize_error_message("") == ""

    def test_no_sensitive_values(self):
        """Test basic message without sensitive values"""
        msg = "Database connection failed"
        assert sanitize_error_message(msg) == msg

    def test_custom_sensitive_values(self):
        """Test removal of custom sensitive values"""
        msg = "Connection failed for user secret123 with password mysecret"
        result = sanitize_error_message(msg, ["secret123", "mysecret"])
        assert "secret123" not in result
        assert "mysecret" not in result
        assert "***" in result

    def test_skip_short_sensitive_values(self):
        """Test that values shorter than 3 chars are skipped"""
        msg = "Error with value ab"
        result = sanitize_error_message(msg, ["ab"])
        assert "ab" in result  # Should not be sanitized

    def test_url_encoded_values(self):
        """Test sanitization of URL-encoded sensitive values"""
        msg = "Failed with encoded value test%40example"
        result = sanitize_error_message(msg, ["test@example"])
        assert "test%40example" not in result
        assert "***" in result

    def test_base64_encoded_values(self):
        """Test sanitization of base64-encoded values"""
        msg = "Token dGVzdA== failed"
        result = sanitize_error_message(msg, ["test"])
        assert "dGVzdA==" not in result
        assert "***" in result

    def test_invalid_base64_handling(self):
        """Test graceful handling of invalid base64 encoding"""
        msg = "Invalid unicode string \\xff"
        result = sanitize_error_message(msg, ["\\xff"])
        # Should not crash and should sanitize the original value
        assert "***" in result

    def test_connection_string_patterns(self):
        """Test sanitization of database connection strings"""
        msg = "Failed to connect to mongodb://user:pass@localhost:27017"
        result = sanitize_error_message(msg)
        assert "user:pass" not in result
        assert "mongodb://***:***@***" in result

    def test_authorization_bearer_patterns(self):
        """Test sanitization of Bearer tokens"""
        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.signature failed"
        result = sanitize_error_message(msg)
        assert "eyJhbGciOiJIUzI1NiJ9.test.signature" not in result
        assert "Authorization: ***" in result

    def test_authorization_basic_patterns(self):
        """Test sanitization of Basic auth"""
        msg = "Authorization: Basic dXNlcjpwYXNz failed"
        result = sanitize_error_message(msg)
        assert "dXNlcjpwYXNz" not in result
        assert "Authorization: ***" in result

    def test_password_key_patterns(self):
        """Test sanitization of password/key patterns"""
        test_cases = [
            ("password: secretvalue", "password: ***"),
            ("api_key=ABC123", "api_key: ***"),
            ('token="xyz789"', "token: ***"),
            ("secret: 'mysecret'", "secret: ***"),
            ("credential=test123", "credential: ***"),
        ]

        for input_msg, expected_pattern in test_cases:
            result = sanitize_error_message(input_msg)
            assert expected_pattern in result
            # Ensure the sensitive value is removed
            sensitive_part = input_msg.split(":")[-1].split("=")[-1].strip("\"'")
            assert sensitive_part not in result

    def test_bearer_token_patterns(self):
        """Test removal of Bearer tokens in various formats"""
        msg = "Bearer ABC123.DEF456.GHI789 authentication failed"
        result = sanitize_error_message(msg)
        assert "ABC123.DEF456.GHI789" not in result
        assert "Bearer ***" in result

    def test_basic_token_patterns(self):
        """Test removal of Basic tokens"""
        msg = "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ== failed"
        result = sanitize_error_message(msg)
        assert "QWxhZGRpbjpvcGVuIHNlc2FtZQ==" not in result
        assert "Basic ***" in result

    def test_json_password_patterns(self):
        """Test sanitization of passwords in JSON format"""
        msg = '{"password": "secretpass", "user": "test"}'
        result = sanitize_error_message(msg)
        assert '"password": "***"' in result
        assert "secretpass" not in result
        assert '"user": "test"' in result  # Non-password fields preserved

    def test_database_protocol_patterns(self):
        """Test various database protocol connection strings"""
        protocols = [
            "postgresql://user:pass@host/db",
            "mysql://user:pass@host/db",
            "redis://user:pass@host:6379",
            "neo4j://user:pass@host:7687",
        ]

        for protocol_url in protocols:
            msg = f"Failed to connect to {protocol_url}"
            result = sanitize_error_message(msg)
            assert "user:pass" not in result
            assert "***:***@***" in result

    def test_bolt_protocol_patterns(self):
        """Test Neo4j Bolt protocol sanitization"""
        test_cases = [
            "bolt://user:pass@localhost:7687",
            "bolt+s://user:pass@localhost:7687",
        ]

        for bolt_url in test_cases:
            msg = f"Bolt connection failed: {bolt_url}"
            result = sanitize_error_message(msg)
            assert "user:pass" not in result
            assert "bolt://***:***@" in result

    def test_case_insensitive_patterns(self):
        """Test that pattern matching is case insensitive"""
        msg = "PASSWORD: SecretValue failed"
        result = sanitize_error_message(msg)
        assert "SecretValue" not in result
        assert "PASSWORD: ***" in result or "password: ***" in result


class TestGetEnvironment:
    """Test environment detection logic"""

    def test_environment_variable(self):
        """Test ENVIRONMENT variable takes precedence"""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert get_environment() == "production"

    def test_env_variable_fallback(self):
        """Test ENV variable as fallback"""
        with patch.dict(os.environ, {"ENV": "staging"}, clear=True):
            assert get_environment() == "staging"

    def test_node_env_fallback(self):
        """Test NODE_ENV variable as final fallback"""
        with patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True):
            assert get_environment() == "production"

    def test_default_development(self):
        """Test default to development when no env vars set"""
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() == "development"

    def test_production_variations(self):
        """Test various production environment names"""
        for env_value in ["prod", "production", "PROD", "PRODUCTION"]:
            with patch.dict(os.environ, {"ENVIRONMENT": env_value}):
                assert get_environment() == "production"

    def test_staging_variations(self):
        """Test various staging environment names"""
        for env_value in ["stage", "staging", "STAGE", "STAGING"]:
            with patch.dict(os.environ, {"ENVIRONMENT": env_value}):
                assert get_environment() == "staging"

    def test_development_variations(self):
        """Test various development environment names"""
        for env_value in ["dev", "development", "test", "local"]:
            with patch.dict(os.environ, {"ENVIRONMENT": env_value}):
                assert get_environment() == "development"


class TestGetSecureConnectionConfig:
    """Test secure connection configuration generation"""

    def test_basic_config_development(self):
        """Test basic config in development environment"""
        config = {"neo4j": {"host": "localhost", "port": 7687}}

        with patch("src.core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "neo4j")

        expected = {
            "host": "localhost",
            "port": 7687,
            "ssl": False,
            "verify_ssl": True,
            "timeout": 30,
            "environment": "development",
        }
        assert result == expected

    def test_basic_config_production(self):
        """Test basic config in production environment"""
        config = {"neo4j": {"host": "prod-server", "port": 7687}}

        with patch("src.core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "neo4j")

        expected = {
            "host": "prod-server",
            "port": 7687,
            "ssl": True,  # Default to True in production
            "verify_ssl": True,
            "timeout": 30,
            "environment": "production",
        }
        assert result == expected

    def test_production_ssl_disabled_warning(self):
        """Test warning when SSL is explicitly disabled in production"""
        config = {"neo4j": {"host": "localhost", "ssl": False}}

        with patch("src.core.utils.get_environment", return_value="production"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                get_secure_connection_config(config, "neo4j")

                assert len(w) == 1
                assert "SSL is disabled" in str(w[0].message)
                assert "security risk" in str(w[0].message)
                assert w[0].category == RuntimeWarning

    def test_custom_ssl_settings(self):
        """Test custom SSL configuration"""
        config = {
            "qdrant": {
                "host": "secure-server",
                "port": 6333,
                "ssl": True,
                "verify_ssl": False,
                "timeout": 60,
            }
        }

        with patch("src.core.utils.get_environment", return_value="staging"):
            result = get_secure_connection_config(config, "qdrant")

        expected = {
            "host": "secure-server",
            "port": 6333,
            "ssl": True,
            "verify_ssl": False,
            "timeout": 60,
            "environment": "staging",
        }
        assert result == expected

    def test_ssl_certificate_paths(self):
        """Test SSL certificate path configuration"""
        config = {
            "neo4j": {
                "host": "secure-neo4j",
                "ssl_cert_path": "/path/to/cert.pem",
                "ssl_key_path": "/path/to/key.pem",
                "ssl_ca_path": "/path/to/ca.pem",
            }
        }

        with patch("src.core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "neo4j")

        assert result["ssl_cert_path"] == "/path/to/cert.pem"
        assert result["ssl_key_path"] == "/path/to/key.pem"
        assert result["ssl_ca_path"] == "/path/to/ca.pem"

    def test_missing_service_config(self):
        """Test handling of missing service configuration"""
        config: dict[str, dict[str, str]] = {}

        with patch("src.core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "missing-service")

        expected = {
            "host": "localhost",
            "port": None,
            "ssl": False,
            "verify_ssl": True,
            "timeout": 30,
            "environment": "development",
        }
        assert result == expected

    def test_partial_ssl_cert_config(self):
        """Test partial SSL certificate configuration"""
        config = {
            "neo4j": {
                "ssl_cert_path": "/path/to/cert.pem",
                # Missing ssl_key_path and ssl_ca_path
            }
        }

        with patch("src.core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "neo4j")

        assert result["ssl_cert_path"] == "/path/to/cert.pem"
        assert "ssl_key_path" not in result
        assert "ssl_ca_path" not in result

    def test_default_port_handling(self):
        """Test default port assignment when not specified"""
        config = {"service": {"host": "example.com"}}

        with patch("src.core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "service")

        assert result["port"] is None  # Should be None when not specified
