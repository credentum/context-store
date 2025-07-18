# Context Store

A standalone MCP-based context storage service providing vector similarity search, graph traversal, and persistent storage for AI agent interactions.

## Overview

Context Store is extracted from the agent-context-template project to provide a clean, standalone service for context management. It implements the Model Context Protocol (MCP) to offer context operations as external tools.

## Features

- **Vector Storage**: Qdrant-based similarity search for semantic context retrieval
- **Graph Storage**: Neo4j-based relationship traversal and context linking  
- **Key-Value Store**: Fast transient storage for agent state and scratchpad data
- **MCP Protocol**: Standard protocol interface for AI agent integration
- **Validation**: Comprehensive YAML schema validation for all context types
- **Docker Ready**: One-command deployment with all dependencies

## Architecture

```
context-store/
├── src/
│   ├── storage/          # Database clients and operations
│   ├── validators/       # Schema validation and data integrity
│   ├── mcp_server/       # MCP protocol server implementation
│   └── core/            # Shared utilities and base classes
├── schemas/             # YAML schemas for context validation
├── contracts/           # MCP tool contracts and specifications
├── tests/              # Test suite
└── docker/             # Docker configuration and deployment
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and start all services
git clone https://github.com/credentum/context-store.git
cd context-store
docker-compose up -d

# Verify services
curl http://localhost:8000/health
```

### Manual Installation

```bash
# Python dependencies
pip install -r requirements.txt

# TypeScript dependencies  
npm install

# Start MCP server
npm run start
```

## MCP Tools

The context store provides these MCP tools:

- `store_context`: Store context data with vector embeddings and graph relationships
- `retrieve_context`: Hybrid retrieval using vector similarity and graph traversal
- `query_graph`: Direct Cypher queries for advanced graph operations
- `update_scratchpad`: Transient key-value storage with TTL
- `get_agent_state`: Retrieve persistent agent memory and state

## Configuration

Configure via environment variables:

```bash
# Database connections
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# MCP server
MCP_SERVER_PORT=8000
MCP_LOG_LEVEL=info

# Storage settings
VECTOR_COLLECTION=context_embeddings
GRAPH_DATABASE=context_graph
```

## Development

### Prerequisites

- Python 3.8+
- Node.js 18+
- Docker and Docker Compose
- Qdrant v1.14.x
- Neo4j Community Edition

### Setup

```bash
# Development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Start development services
docker-compose -f docker-compose.dev.yml up -d
```

### Testing

```bash
# Run all tests
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/

# Run MCP tool tests
npm test
```

## Deployment

### Local Development

```bash
docker-compose up -d
```

### Production

```bash
# Using production configuration
docker-compose -f docker-compose.prod.yml up -d

# Scale MCP servers
docker-compose up --scale mcp-server=3
```

### Cloud Deployment

See `docs/deployment/` for cloud-specific deployment guides:

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Kubernetes

## API Documentation

### Health Check

```bash
GET /health
```

Returns service status and dependency health.

### MCP Protocol

The server implements the MCP specification. Connect using any MCP-compatible client:

```typescript
import { MCPClient } from '@modelcontextprotocol/client';

const client = new MCPClient({
  serverUrl: 'http://localhost:8000/mcp'
});

await client.connect();
const result = await client.callTool('store_context', {
  type: 'design',
  content: { ... },
  metadata: { ... }
});
```

## Performance

- **Response Time**: <50ms for typical MCP tool calls
- **Throughput**: 1000+ concurrent connections supported
- **Storage**: Scales with Qdrant and Neo4j limits
- **Memory**: ~100MB base memory usage

## Security

- **Authentication**: API key authentication for database access
- **Authorization**: Role-based access control for graph queries
- **Input Validation**: Comprehensive schema validation
- **Network**: TLS encryption for all external connections

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/credentum/context-store/issues)
- **Discussions**: [GitHub Discussions](https://github.com/credentum/context-store/discussions)

## Related Projects

- [agent-context-template](https://github.com/credentum/agent-context-template) - Reference implementation using context-store
- [MCP Specification](https://github.com/modelcontextprotocol/specification) - Model Context Protocol documentation