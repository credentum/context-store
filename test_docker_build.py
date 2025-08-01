#!/usr/bin/env python3
"""
Quick test to verify the extracted components can be imported
"""

try:
    print("Testing imports...")
    
    # Test storage imports
    from src.storage.hash_diff_embedder import HashDiffEmbedder, DocumentHash, EmbeddingTask
    print("✓ hash_diff_embedder imports successful")
    
    from src.storage.neo4j_client import Neo4jInitializer
    print("✓ neo4j_client imports successful")
    
    from src.storage.qdrant_client import VectorDBInitializer
    print("✓ qdrant_client imports successful")
    
    from src.storage.kv_store import ContextKV
    print("✓ kv_store imports successful")
    
    # Test core imports
    from src.core.base_component import DatabaseComponent
    print("✓ base_component imports successful")
    
    from src.core.utils import get_secure_connection_config
    print("✓ utils imports successful")
    
    # Test validator imports
    from src.validators.kv_validators import validate_redis_key
    print("✓ validators imports successful")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    exit(1)