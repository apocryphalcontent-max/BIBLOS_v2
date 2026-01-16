"""
BIBLOS v2 - Service Factories

Provides factory classes and functions for creating complex objects
with proper dependency injection and configuration.

Design Patterns:
    - Abstract Factory: Creates families of related objects
    - Factory Method: Defers instantiation to subclasses
    - Builder: Step-by-step construction of complex objects

Usage:
    from core.factories import PipelineFactory, AgentFactory

    # Create a configured pipeline
    pipeline = await PipelineFactory.create_production_pipeline()

    # Create an agent with dependencies
    agent = await AgentFactory.create_agent("PATROLOGOS", db_client=db)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

logger = logging.getLogger("biblos.factories")

T = TypeVar("T")


# =============================================================================
# BASE FACTORY INTERFACES
# =============================================================================


class Factory(ABC, Generic[T]):
    """Abstract factory interface."""

    @abstractmethod
    async def create(self, **kwargs: Any) -> T:
        """Create an instance with the given parameters."""
        pass


class Builder(ABC, Generic[T]):
    """Abstract builder interface for step-by-step construction."""

    @abstractmethod
    def reset(self) -> "Builder[T]":
        """Reset the builder to initial state."""
        pass

    @abstractmethod
    def build(self) -> T:
        """Build and return the final product."""
        pass


# =============================================================================
# PIPELINE FACTORY
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for pipeline creation."""
    enable_linguistic: bool = True
    enable_theological: bool = True
    enable_intertextual: bool = True
    enable_validation: bool = True
    parallel_execution: bool = True
    max_parallel_agents: int = 8
    phase_timeout_seconds: float = 300.0
    enable_event_sourcing: bool = True
    enable_circuit_breakers: bool = True


class PipelineFactory:
    """
    Factory for creating configured pipeline instances.

    Creates pipelines with proper phase configuration, database connections,
    and ML component wiring.
    """

    @staticmethod
    async def create_default() -> Any:
        """Create a pipeline with default configuration."""
        return await PipelineFactory.create(PipelineConfig())

    @staticmethod
    async def create_production() -> Any:
        """Create a production-ready pipeline with all features enabled."""
        config = PipelineConfig(
            enable_linguistic=True,
            enable_theological=True,
            enable_intertextual=True,
            enable_validation=True,
            parallel_execution=True,
            max_parallel_agents=8,
            phase_timeout_seconds=300.0,
            enable_event_sourcing=True,
            enable_circuit_breakers=True,
        )
        return await PipelineFactory.create(config)

    @staticmethod
    async def create_development() -> Any:
        """Create a development pipeline with debugging features."""
        config = PipelineConfig(
            enable_linguistic=True,
            enable_theological=True,
            enable_intertextual=True,
            enable_validation=False,  # Skip validation for faster iteration
            parallel_execution=False,  # Sequential for easier debugging
            max_parallel_agents=1,
            phase_timeout_seconds=600.0,  # Longer timeout for debugging
            enable_event_sourcing=False,
            enable_circuit_breakers=False,
        )
        return await PipelineFactory.create(config)

    @staticmethod
    async def create_testing() -> Any:
        """Create a minimal pipeline for testing."""
        config = PipelineConfig(
            enable_linguistic=True,
            enable_theological=False,
            enable_intertextual=False,
            enable_validation=False,
            parallel_execution=False,
            max_parallel_agents=1,
            phase_timeout_seconds=30.0,
            enable_event_sourcing=False,
            enable_circuit_breakers=False,
        )
        return await PipelineFactory.create(config)

    @staticmethod
    async def create(config: PipelineConfig) -> Any:
        """Create a pipeline with custom configuration."""
        from pipeline.unified_orchestrator import UnifiedOrchestrator

        # Build phases based on config
        phases = []

        if config.enable_linguistic:
            from pipeline.phases.linguistic import LinguisticPhase
            phases.append(LinguisticPhase)

        if config.enable_theological:
            from pipeline.phases.theological import TheologicalPhase
            phases.append(TheologicalPhase)

        if config.enable_intertextual:
            from pipeline.phases.intertextual import IntertextualPhase
            from pipeline.phases.cross_reference import CrossReferencePhase
            phases.append(IntertextualPhase)
            phases.append(CrossReferencePhase)

        if config.enable_validation:
            from pipeline.phases.validation import ValidationPhase
            phases.append(ValidationPhase)

        logger.info(f"Creating pipeline with {len(phases)} phases")

        orchestrator = UnifiedOrchestrator()
        # Configure based on settings
        return orchestrator


# =============================================================================
# AGENT FACTORY
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for agent creation."""
    name: str
    extraction_type: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    enable_caching: bool = True
    enable_tracing: bool = True
    llm_model: Optional[str] = None
    temperature: float = 0.0


class AgentFactory:
    """
    Factory for creating configured extraction agents.

    Handles agent instantiation with proper dependency injection
    for database clients, ML models, and observability.
    """

    # Registry of agent types to their classes
    _agent_registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type) -> None:
        """Register an agent class."""
        cls._agent_registry[name.upper()] = agent_class
        logger.debug(f"Registered agent: {name}")

    @classmethod
    def get_available_agents(cls) -> List[str]:
        """Get list of available agent names."""
        return list(cls._agent_registry.keys())

    @classmethod
    async def create(cls, config: AgentConfig, **dependencies: Any) -> Any:
        """
        Create an agent with the given configuration and dependencies.

        Args:
            config: Agent configuration
            dependencies: Injected dependencies (db_client, vector_store, etc.)

        Returns:
            Configured agent instance
        """
        agent_class = cls._agent_registry.get(config.name.upper())
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {config.name}")

        # Create agent with injected dependencies
        agent = agent_class(**dependencies)

        logger.info(f"Created agent: {config.name}")
        return agent

    @classmethod
    async def create_linguistic_suite(cls, **dependencies: Any) -> Dict[str, Any]:
        """Create all linguistic analysis agents."""
        agents = {}
        linguistic_agents = [
            "GRAMMATEUS", "MORPHOLOGOS", "SYNTAKTIKOS",
            "SEMANTIKOS", "LEXIKOS", "ETYMOLOGOS"
        ]
        for name in linguistic_agents:
            if name in cls._agent_registry:
                config = AgentConfig(name=name, extraction_type="linguistic")
                agents[name] = await cls.create(config, **dependencies)
        return agents

    @classmethod
    async def create_theological_suite(cls, **dependencies: Any) -> Dict[str, Any]:
        """Create all theological analysis agents."""
        agents = {}
        theological_agents = [
            "PATROLOGOS", "TYPOLOGOS", "THEOLOGOS",
            "LITURGIKOS", "DOGMATIKOS"
        ]
        for name in theological_agents:
            if name in cls._agent_registry:
                config = AgentConfig(name=name, extraction_type="theological")
                agents[name] = await cls.create(config, **dependencies)
        return agents


# =============================================================================
# DATABASE CLIENT FACTORY
# =============================================================================


@dataclass
class DatabaseConfig:
    """Configuration for database client creation."""
    host: str = "localhost"
    port: int = 5432
    database: str = "biblos_v2"
    user: str = "biblos"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    timeout_seconds: int = 30


class DatabaseClientFactory:
    """Factory for creating database clients."""

    @staticmethod
    async def create_postgres(config: Optional[DatabaseConfig] = None) -> Any:
        """Create a PostgreSQL client."""
        from db.postgres_optimized import PostgresClient

        config = config or DatabaseConfig()
        client = PostgresClient(
            database_url=f"postgresql+asyncpg://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}",
            pool_size=config.pool_size,
            max_overflow=config.max_overflow
        )
        await client.initialize()
        return client

    @staticmethod
    async def create_neo4j(
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = ""
    ) -> Any:
        """Create a Neo4j client."""
        from db.neo4j_optimized import Neo4jClient, Neo4jConfig

        config = Neo4jConfig(uri=uri, user=user, password=password)
        client = Neo4jClient(config=config)
        await client.connect()
        return client

    @staticmethod
    async def create_qdrant(
        host: str = "localhost",
        port: int = 6333
    ) -> Any:
        """Create a Qdrant vector store client."""
        from db.qdrant_client import QdrantVectorStore

        client = QdrantVectorStore(host=host, port=port)
        await client.connect()
        return client


# =============================================================================
# ML ENGINE FACTORY
# =============================================================================


@dataclass
class MLEngineConfig:
    """Configuration for ML engine creation."""
    device: str = "cuda"
    model_path: Optional[str] = None
    batch_size: int = 32
    enable_fp16: bool = True


class MLEngineFactory:
    """Factory for creating ML engine instances."""

    @staticmethod
    async def create_embedding_model(
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda"
    ) -> Any:
        """Create an embedding model."""
        from ml.embeddings.domain_embedders import MultiDomainEmbedder

        embedder = MultiDomainEmbedder(model_name=model_name, device=device)
        await embedder.initialize()
        return embedder

    @staticmethod
    async def create_gnn_model(config: Optional[MLEngineConfig] = None) -> Any:
        """Create a GNN model for cross-reference prediction."""
        config = config or MLEngineConfig()
        from ml.models.gnn import GNNCrossRefPredictor

        model = GNNCrossRefPredictor(device=config.device)
        return model

    @staticmethod
    async def create_inference_pipeline(
        embedding_model: Any,
        gnn_model: Any,
        vector_store: Any
    ) -> Any:
        """Create the full inference pipeline."""
        from ml.inference.pipeline import InferencePipeline, InferenceConfig

        config = InferenceConfig()
        pipeline = InferencePipeline(config)
        await pipeline.initialize()
        return pipeline


# =============================================================================
# GOLDEN RECORD BUILDER
# =============================================================================


class GoldenRecordBuilder:
    """
    Builder for constructing Golden Record entries step by step.

    Usage:
        record = (GoldenRecordBuilder()
            .with_verse("GEN.1.1")
            .with_linguistic_data(linguistic_results)
            .with_theological_data(theological_results)
            .with_cross_references(crossrefs)
            .with_confidence(0.95)
            .build())
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "GoldenRecordBuilder":
        """Reset the builder to initial state."""
        self._verse_id: Optional[str] = None
        self._linguistic: Dict[str, Any] = {}
        self._theological: Dict[str, Any] = {}
        self._intertextual: Dict[str, Any] = {}
        self._cross_references: List[Dict] = []
        self._confidence: float = 0.0
        self._quality_tier: int = 0
        self._metadata: Dict[str, Any] = {}
        return self

    def with_verse(self, verse_id: str) -> "GoldenRecordBuilder":
        """Set the verse ID."""
        self._verse_id = verse_id
        return self

    def with_linguistic_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add linguistic analysis data."""
        self._linguistic = data
        return self

    def with_theological_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add theological analysis data."""
        self._theological = data
        return self

    def with_intertextual_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add intertextual analysis data."""
        self._intertextual = data
        return self

    def with_cross_references(self, crossrefs: List[Dict]) -> "GoldenRecordBuilder":
        """Add cross-references."""
        self._cross_references = crossrefs
        return self

    def with_confidence(self, confidence: float) -> "GoldenRecordBuilder":
        """Set overall confidence score."""
        self._confidence = confidence
        return self

    def with_quality_tier(self, tier: int) -> "GoldenRecordBuilder":
        """Set quality certification tier."""
        self._quality_tier = tier
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add metadata."""
        self._metadata = metadata
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the final Golden Record."""
        if not self._verse_id:
            raise ValueError("verse_id is required")

        record = {
            "verse_id": self._verse_id,
            "linguistic": self._linguistic,
            "theological": self._theological,
            "intertextual": self._intertextual,
            "cross_references": self._cross_references,
            "confidence": self._confidence,
            "quality_tier": self._quality_tier,
            "metadata": self._metadata,
        }

        # Reset for next use
        self.reset()
        return record
