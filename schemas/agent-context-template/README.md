# Agent Context Template Schemas

This directory contains YAML schemas specifically designed for the Agent Context Template system. These schemas provide structured validation for different types of context documents used in AI agent workflows.

## Schema Overview

### Core Schemas

- **`base.yaml`** - Base schema for all context documents with common fields
- **`decision.yaml`** - Simple decision tracking schema
- **`design.yaml`** - Design document schema for system architecture
- **`log.yaml`** - Log entry schema for execution traces
- **`sprint.yaml`** - Sprint planning and tracking schema
- **`trace.yaml`** - Execution trace schema for debugging

### Extended Schemas

- **`decision_full.yaml`** - Comprehensive decision tracking with context
- **`design_full.yaml`** - Detailed design schema with dependencies
- **`log_full.yaml`** - Full log schema with metadata and stack traces
- **`sprint_full.yaml`** - Complete sprint schema with metrics
- **`trace_full.yaml`** - Detailed trace schema with performance data

## Schema Loader

The **`schema_loader.py`** module provides utilities for loading and validating documents against these schemas programmatically.

## Usage

### Python Integration

```python
from schemas.agent_context_template.schema_loader import SchemaLoader

# Load a schema
loader = SchemaLoader()
schema = loader.load_schema('decision')

# Validate a document
document = {
    "decision_id": "decide-001",
    "title": "Database Selection",
    "status": "approved"
}

is_valid, errors = loader.validate_document(document, 'decision')
```

### Direct YAML Validation

```bash
# Using yamllint with these schemas
yamllint -c schemas/agent-context-template/base.yaml my-document.yaml
```

## Schema Design Principles

1. **Modularity**: Each schema serves a specific purpose in the agent workflow
2. **Extensibility**: Base schema provides common fields, specialized schemas extend
3. **Validation**: Strong type checking and required field validation
4. **Documentation**: All schemas include descriptions and examples
5. **Compatibility**: Designed to work with existing MCP tools and storage systems

## Integration with Context Store

These schemas are designed to integrate seamlessly with the context-store's:

- **Vector Storage**: Document embeddings with schema validation
- **Graph Storage**: Relationship tracking between schema-validated documents
- **MCP Protocol**: Schema-aware tool contracts
- **Validation Pipeline**: Automatic schema validation on document ingestion

## Contributing

When adding new schemas:

1. Follow the existing naming convention (`type.yaml` and `type_full.yaml`)
2. Extend from `base.yaml` when possible
3. Include comprehensive field descriptions
4. Add validation examples
5. Update the schema loader module
6. Document integration points

## Examples

See the individual schema files for detailed examples and field descriptions. Each schema includes:

- Field definitions with types and constraints
- Required vs optional field specifications
- Example documents that validate against the schema
- Integration notes for context store usage