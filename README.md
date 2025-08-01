# Context Store

Core validation and utility modules for the Agent Context Template system.

## Overview

This package provides clean, agent-independent validators and utilities that can be used across the context system. It includes:

- **Configuration Validators**: Validate YAML configuration files for correctness and security
- **KV Validators**: Input validation for key-value store operations
- **Core Utilities**: Common utility functions including error sanitization and secure connection configuration
- **YAML Schemas**: Schema definitions for context documents

## Installation

```bash
pip install -e .
```

## Usage

### Configuration Validation

```python
from src.validators import ConfigValidator

validator = ConfigValidator()
is_valid, errors, warnings = validator.validate_all()

if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### KV Store Validation

```python
from src.validators import (
    validate_cache_entry,
    validate_metric_event,
    validate_redis_key
)

# Validate cache entry
cache_data = {
    "key": "user:123",
    "value": {"name": "John"},
    "created_at": "2025-01-01T00:00:00",
    "ttl_seconds": 3600
}
is_valid = validate_cache_entry(cache_data)

# Validate Redis key
is_valid = validate_redis_key("my:redis:key")
```

### Core Utilities

```python
from src.core import (
    sanitize_error_message,
    get_environment,
    get_secure_connection_config
)

# Sanitize error messages
safe_error = sanitize_error_message(
    "Connection failed: user:password@host",
    sensitive_values=["password"]
)

# Get environment
env = get_environment()  # Returns: 'development', 'staging', or 'production'

# Get secure connection config
config = get_secure_connection_config(config_dict, "neo4j")
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Project Structure

```
context-store/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── utils.py
│   └── validators/
│       ├── __init__.py
│       ├── config_validator.py
│       └── kv_validators.py
├── schemas/
│   ├── base.yaml
│   ├── decision.yaml
│   ├── design.yaml
│   ├── log.yaml
│   ├── sprint.yaml
│   ├── trace.yaml
│   └── schema_loader.py
├── tests/
│   ├── test_config_validator.py
│   ├── test_core_utils_comprehensive.py
│   ├── test_kv_validators.py
│   └── test_kv_validators_comprehensive.py
├── setup.py
└── README.md
```

## License

MIT License - See LICENSE file in the root repository.