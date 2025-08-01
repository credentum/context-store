#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for storage tests
"""

import os
import pytest


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return {
        "qdrant": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "api_key": os.getenv("QDRANT_API_KEY", "test_key"),
            "timeout": 5,
            "collection_name": "test_collection",
            "embedding_model": "text-embedding-ada-002",
            "embedding_dim": 1536,
            "verify_ssl": False,
        },
        "neo4j": {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "test_password"),
            "database": "test_db",
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": 0,
            "decode_responses": True,
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", "test_openai_key"),
            "organization": os.getenv("OPENAI_ORGANIZATION", None),
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
        },
    }