# CLAUDE.md

This file provides guidance for Claude Code, Gemini CLI, GitHub Copilot CLI, and other AI coding assistants when working with the BIBLOS v2 codebase.

## Project Overview

BIBLOS v2 is a comprehensive ML pipeline orchestration system for biblical scholarship, featuring:
- 24-agent SDES (Scripture Data Extraction System) pipeline
- Cross-reference discovery using GNN and embedding models
- Orthodox Christian patristic integration
- LangChain/LangGraph multi-agent framework

## Quick Start Commands

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .
pip install -e ".[dev]"    # Include dev dependencies
```

### CLI Commands
```bash
# Check system status
biblos status

# Process a single verse
biblos process --verse "GEN.1.1"

# Run batch processing
biblos batch --book "GEN" --start-chapter 1 --end-chapter 3

# Discover cross-references
biblos discover --verse "GEN.1.1" --top-k 10

# Export results
biblos export --book "GEN" --format json --output ./output

# Validate data
biblos validate --full

# Start API server
biblos serve --port 8000
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=ml --cov=pipeline

# Run specific test category
pytest tests/agents/ -v
pytest -m "not slow"       # Skip slow tests
pytest -m integration      # Only integration tests
```

## Architecture

### Directory Structure
```
BIBLOS_v2/
├── agents/                 # 24 SDES extraction agents
│   ├── linguistic/         # Grammatical analysis agents
│   ├── theological/        # Patristic & typological agents
│   ├── intertextual/       # Cross-reference agents
│   └── validation/         # Quality assurance agents
├── api/                    # FastAPI REST endpoints
├── cli/                    # Typer CLI application
├── config.py               # Centralized configuration
├── data/                   # Dataset classes & schemas
│   ├── schemas.py          # Normalized data schemas
│   ├── dataset.py          # PyTorch datasets
│   └── loaders.py          # DataLoader factories
├── db/                     # Database clients
│   ├── neo4j_client.py     # Graph database
│   ├── postgres_client.py  # Relational storage
│   └── redis_client.py     # Caching layer
├── integrations/           # External corpus integrations
│   ├── text_fabric.py      # Text-Fabric (BHSA, SBLGNT)
│   └── macula.py           # Macula Greek/Hebrew
├── ml/                     # Machine learning components
│   ├── embeddings/         # Sentence embeddings
│   ├── models/             # GNN & classifiers
│   └── inference/          # Inference pipeline
├── pipeline/               # Pipeline orchestration
│   ├── orchestrator.py     # Main pipeline coordinator
│   └── phases/             # Phase implementations
├── scripts/                # Data population utilities
└── tests/                  # Test suite
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `agents/base.py` | BaseExtractionAgent with LangChain integration |
| `pipeline/orchestrator.py` | 4-phase pipeline coordinator |
| `ml/inference/pipeline.py` | Cross-reference discovery |
| `data/schemas.py` | Normalized data schemas |
| `config.py` | Environment-based configuration |

## Data Schemas

All data types use centralized schemas from `data/schemas.py`:
- `VerseSchema` - Bible verses
- `WordSchema` - Word-level analysis
- `CrossReferenceSchema` - Cross-reference connections
- `ExtractionResultSchema` - Agent extraction results
- `GoldenRecordSchema` - Final pipeline output

## Environment Configuration

Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Key variables:
- `OPENAI_API_KEY` - For LLM-powered agents
- `NEO4J_URI` - Graph database connection
- `POSTGRES_*` - PostgreSQL settings
- `ML_DEVICE` - `cuda` or `cpu`

## Common Tasks

### Adding a New Agent
1. Create agent in appropriate directory (e.g., `agents/theological/`)
2. Inherit from `BaseExtractionAgent`
3. Implement `extract()`, `validate()`, `get_dependencies()`
4. Register in phase configuration (`config.py`)

### Adding Cross-References
Use the normalized schema:
```python
from data.schemas import CrossReferenceSchema

ref = CrossReferenceSchema(
    source_ref="GEN.1.1",
    target_ref="JHN.1.1",
    connection_type="typological",
    strength="strong",
    confidence=0.95
)
```

### Running Inference
```python
from ml.inference.pipeline import InferencePipeline, InferenceConfig

config = InferenceConfig(min_confidence=0.7)
pipeline = InferencePipeline(config)
await pipeline.initialize()
result = await pipeline.infer("GEN.1.1", "In the beginning...")
```

## Code Standards

- Python 3.11+
- Type hints required
- Line length: 88 (Black)
- Formatting: `black` and `ruff`
- Testing: pytest with 100% coverage goal
- Unicode: UTF-8 for Greek/Hebrew text

## Connection Types

Valid cross-reference types:
- `typological` - Type/antitype relationships
- `prophetic` - Prophecy fulfillment
- `verbal` - Shared vocabulary
- `thematic` - Shared themes
- `conceptual` - Related concepts
- `historical` - Historical connections
- `liturgical` - Worship connections
- `narrative` - Story parallels
- `genealogical` - Family lineages
- `geographical` - Location references

## MCP Server Integration

For Claude Code users, this project works with:
- `memory` - Persistent context
- `sequential-thinking` - Complex analysis
- `filesystem` - File operations
- `context7` - Documentation lookups

## Orthodox Theological Context

All theological analysis respects Orthodox Christian tradition:
- Septuagint (LXX) priority alongside Masoretic text
- Patristic sources (Chrysostom, Basil, Gregory, Cyril, Augustine)
- Fourfold sense: literal, allegorical, tropological, anagogical
- Typological connections between OT and NT
