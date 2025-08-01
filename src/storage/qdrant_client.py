#!/usr/bin/env python3
"""
qdrant_client.py: Qdrant client for context storage system

This module:
1. Manages Qdrant vector database connections
2. Creates and configures collections
3. Handles vector operations
4. Provides search and indexing capabilities
"""

import sys
import time
from typing import Any, Dict, Optional

import click
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)


class VectorDBInitializer:
    """Initialize and configure Qdrant vector database"""

    def __init__(self, config_path: str = ".ctxrc.yaml"):
        self.config = self._load_config(config_path)
        self.client: Optional[QdrantClient] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from .ctxrc.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    click.echo(f"Error: {config_path} must contain a dictionary", err=True)
                    sys.exit(1)
                return config
        except FileNotFoundError:
            click.echo(f"Error: {config_path} not found", err=True)
            sys.exit(1)
        except yaml.YAMLError as e:
            click.echo(f"Error parsing {config_path}: {e}", err=True)
            sys.exit(1)

    def connect(self) -> bool:
        """Connect to Qdrant instance"""
        # Import locally
        from src.core.utils import get_secure_connection_config

        qdrant_config = get_secure_connection_config(self.config, "qdrant")
        host = qdrant_config["host"]
        port = qdrant_config.get("port", 6333)
        use_ssl = qdrant_config.get("ssl", False)
        timeout = qdrant_config.get("timeout", 5)

        try:
            # Use appropriate protocol based on SSL setting
            if use_ssl:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    https=True,
                    verify=qdrant_config.get("verify_ssl", True),
                    timeout=timeout,
                )
            else:
                self.client = QdrantClient(host=host, port=port, timeout=timeout)
            # Test connection
            if self.client:
                self.client.get_collections()
                click.echo(f"✓ Connected to Qdrant at {host}:{port}")
                return True
            return False
        except Exception as e:
            click.echo(f"✗ Failed to connect to Qdrant at {host}:{port}: {e}", err=True)
            return False

    def create_collection(self, force: bool = False) -> bool:
        """Create the project_context collection"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "project_context")

        if not self.client:
            click.echo("✗ Not connected to Qdrant", err=True)
            return False

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists and not force:
                click.echo(
                    f"Collection '{collection_name}' already exists. Use --force to recreate."
                )
                return True

            if exists and force:
                click.echo(f"Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
                time.sleep(1)  # Give Qdrant time to process

            # Create collection with optimal settings for embeddings
            click.echo(f"Creating collection '{collection_name}'...")

            # Using text-embedding-ada-002 dimensions (1536)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI ada-002 embedding size
                    distance=Distance.COSINE,
                ),
                optimizers_config=OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=2,
                    flush_interval_sec=5,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    full_scan_threshold=10000,
                ),
            )

            click.echo(f"✓ Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            click.echo(f"✗ Failed to create collection: {e}", err=True)
            return False

    def verify_setup(self) -> bool:
        """Verify the Qdrant setup is correct"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "project_context")

        if not self.client:
            click.echo("✗ Not connected to Qdrant", err=True)
            return False

        try:
            # Get collection info
            info = self.client.get_collection(collection_name)

            click.echo("\nCollection Info:")
            click.echo(f"  Name: {collection_name}")

            # Handle different vector config formats
            if info.config and info.config.params and info.config.params.vectors:
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, VectorParams):
                    click.echo(f"  Vector size: {vectors_config.size}")
                    click.echo(f"  Distance metric: {vectors_config.distance}")
                elif isinstance(vectors_config, dict):
                    # Handle named vectors
                    for name, params in vectors_config.items():
                        click.echo(f"  Vector '{name}' size: {params.size}")
                        click.echo(f"  Vector '{name}' distance: {params.distance}")

            click.echo(f"  Points count: {info.points_count}")

            # Check Qdrant version
            qdrant_version = self.config.get("qdrant", {}).get("version", "1.14.x")
            click.echo(f"\nExpected Qdrant version: {qdrant_version}")

            return True

        except Exception as e:
            click.echo(f"✗ Failed to verify setup: {e}", err=True)
            return False

    def insert_test_point(self) -> bool:
        """Insert a test point to verify everything works"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "project_context")

        if not self.client:
            click.echo("✗ Not connected to Qdrant", err=True)
            return False

        try:
            # Create a test embedding (random for now)
            import random

            test_vector = [random.random() for _ in range(1536)]

            # Insert test point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id="test-point-001",
                        vector=test_vector,
                        payload={
                            "document_type": "test",
                            "content": "This is a test point for verification",
                            "created_date": "2025-07-11",
                        },
                    )
                ],
            )

            # Search for it
            results = self.client.search(
                collection_name=collection_name, query_vector=test_vector, limit=1
            )

            if results and results[0].id == "test-point-001":
                click.echo("✓ Test point inserted and retrieved successfully")

                # Clean up
                self.client.delete(
                    collection_name=collection_name, points_selector=["test-point-001"]
                )
                return True
            else:
                click.echo("✗ Test point verification failed", err=True)
                return False

        except Exception as e:
            click.echo(f"✗ Failed to test point operations: {e}", err=True)
            return False


@click.command()
@click.option("--force", is_flag=True, help="Force recreation of collection if exists")
@click.option("--skip-test", is_flag=True, help="Skip test point insertion")
def main(force: bool, skip_test: bool):
    """Initialize Qdrant vector database for the Agent-First Context System"""
    click.echo("=== Qdrant Vector Database Initialization ===\n")

    initializer = VectorDBInitializer()

    # Connect to Qdrant
    if not initializer.connect():
        click.echo("\nPlease ensure Qdrant is running:")
        click.echo("  docker run -p 6333:6333 qdrant/qdrant:v1.14.0")
        sys.exit(1)

    # Create collection
    if not initializer.create_collection(force=force):
        sys.exit(1)

    # Verify setup
    if not initializer.verify_setup():
        sys.exit(1)

    # Test operations
    if not skip_test:
        if not initializer.insert_test_point():
            sys.exit(1)

    click.echo("\n✓ Qdrant initialization complete!")


if __name__ == "__main__":
    main()
