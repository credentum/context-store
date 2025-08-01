#!/usr/bin/env python3
"""
Schema loader utility with backward compatibility for multi-document YAML.
Supports both legacy multi-document format and new single-document format.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def load_schema(filepath: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load schema file with support for both multi-document and single-document formats.

    Args:
        filepath: Path to the schema YAML file

    Returns:
        Dictionary containing the loaded schema or list of documents for multi-doc
    """
    # Add file size limit to prevent DoS (10MB max for schema files)
    MAX_SCHEMA_SIZE = 10 * 1024 * 1024  # 10MB

    try:
        file_path = Path(filepath)
        file_size = file_path.stat().st_size

        if file_size > MAX_SCHEMA_SIZE:
            raise ValueError(
                f"Schema file too large: {file_size} bytes (max: {MAX_SCHEMA_SIZE})"
            )

        with open(filepath, "r") as f:
            content = f.read()
    except IOError as e:
        warnings.warn(f"Failed to read schema file {filepath}: {e}")
        raise

    # Check if it's a multi-document YAML (legacy format)
    if "\n---\n" in content or (
        content.startswith("---\n") and content.count("\n---\n") > 0
    ):
        warnings.warn(
            f"Schema file {filepath} uses deprecated multi-document format. "
            "Please migrate to single-document format.",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(yaml.safe_load_all(content))
    else:
        # New single-document format
        return yaml.safe_load(content)


def convert_multi_to_single(filepath: str) -> Dict[str, Any]:
    """
    Convert a multi-document Yamale schema to single-document format.

    Args:
        filepath: Path to the multi-document schema file

    Returns:
        Dictionary with converted single-document schema
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split by document separator
    docs = list(yaml.safe_load_all(content))

    if len(docs) == 1:
        return docs[0]

    # For Yamale schemas, the first document is the main schema
    # and subsequent documents are sub-schema definitions
    main_schema = docs[0] if docs else {}

    # If there are sub-schemas, add them under a _definitions key
    if len(docs) > 1 and docs[1]:
        main_schema["_definitions"] = docs[1]

    return main_schema


def validate_schema_format(filepath: str) -> bool:
    """
    Validate that a schema file uses the new single-document format.

    Args:
        filepath: Path to the schema YAML file

    Returns:
        True if using new format, False if using legacy format
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Check for multi-document indicators
    has_separator = "\n---\n" in content
    starts_with_separator = content.startswith("---\n")

    # It's multi-doc if it has separators in the middle or starts with --- and has more
    return not (
        has_separator or (starts_with_separator and content.count("\n---\n") > 0)
    )


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        schema = load_schema(sys.argv[1])
        print(f"Loaded schema type: {type(schema)}")
        print(f"Format valid (single-doc): {validate_schema_format(sys.argv[1])}")

        if isinstance(schema, list):
            print(f"Number of documents: {len(schema)}")
        else:
            print(
                f"Top-level keys: "
                f"{list(schema.keys()) if isinstance(schema, dict) else 'N/A'}"
            )
