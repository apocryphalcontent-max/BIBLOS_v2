"""
BIBLOS v2 - Configuration

Centralized configuration management for the entire system.
Uses environment variables with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
import json
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Neo4j settings
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "biblos"))

    # PostgreSQL settings
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "biblos"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "password"))
    postgres_database: str = field(default_factory=lambda: os.getenv("POSTGRES_DATABASE", "biblos"))

    # Redis settings
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))

    # Connection pool settings
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")))

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@dataclass
class MLConfig:
    """Machine learning configuration."""
    # Device settings
    device: str = field(default_factory=lambda: os.getenv("ML_DEVICE", "cuda"))
    use_fp16: bool = field(default_factory=lambda: os.getenv("ML_USE_FP16", "true").lower() == "true")

    # Model paths
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "./models")))
    embeddings_dir: Path = field(default_factory=lambda: Path(os.getenv("EMBEDDINGS_DIR", "./embeddings")))
    checkpoints_dir: Path = field(default_factory=lambda: Path(os.getenv("CHECKPOINTS_DIR", "./checkpoints")))

    # Embedding model settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))
    max_seq_length: int = field(default_factory=lambda: int(os.getenv("MAX_SEQ_LENGTH", "512")))

    # Training settings
    batch_size: int = field(default_factory=lambda: int(os.getenv("ML_BATCH_SIZE", "32")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("ML_LEARNING_RATE", "1e-5")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("ML_NUM_EPOCHS", "10")))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv("ML_WARMUP_STEPS", "1000")))

    # Inference settings
    inference_batch_size: int = field(default_factory=lambda: int(os.getenv("ML_INFERENCE_BATCH_SIZE", "64")))
    max_candidates: int = field(default_factory=lambda: int(os.getenv("ML_MAX_CANDIDATES", "100")))
    min_confidence: float = field(default_factory=lambda: float(os.getenv("ML_MIN_CONFIDENCE", "0.5")))


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration."""
    # Phase settings
    parallel_phases: bool = field(default_factory=lambda: os.getenv("PIPELINE_PARALLEL", "true").lower() == "true")
    max_parallel_agents: int = field(default_factory=lambda: int(os.getenv("PIPELINE_MAX_AGENTS", "4")))
    phase_timeout: int = field(default_factory=lambda: int(os.getenv("PIPELINE_PHASE_TIMEOUT", "300")))

    # Quality thresholds
    min_confidence: float = field(default_factory=lambda: float(os.getenv("PIPELINE_MIN_CONFIDENCE", "0.7")))
    min_coverage: float = field(default_factory=lambda: float(os.getenv("PIPELINE_MIN_COVERAGE", "0.9")))
    gold_threshold: float = field(default_factory=lambda: float(os.getenv("PIPELINE_GOLD_THRESHOLD", "0.9")))
    silver_threshold: float = field(default_factory=lambda: float(os.getenv("PIPELINE_SILVER_THRESHOLD", "0.75")))
    bronze_threshold: float = field(default_factory=lambda: float(os.getenv("PIPELINE_BRONZE_THRESHOLD", "0.5")))

    # Checkpointing
    checkpoint_interval: int = field(default_factory=lambda: int(os.getenv("PIPELINE_CHECKPOINT_INTERVAL", "100")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("PIPELINE_MAX_RETRIES", "3")))


@dataclass
class LLMConfig:
    """Language model configuration for agents."""
    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    openai_temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    openai_max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "4096")))

    # Anthropic settings
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"))

    # Local model settings
    local_model_path: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL_PATH", ""))
    use_local_model: bool = field(default_factory=lambda: os.getenv("USE_LOCAL_MODEL", "false").lower() == "true")


@dataclass
class IntegrationConfig:
    """External integration configuration."""
    # Text-Fabric settings
    text_fabric_data_dir: Path = field(default_factory=lambda: Path(os.getenv("TF_DATA_DIR", "./corpora/tf")))
    text_fabric_cache_dir: Path = field(default_factory=lambda: Path(os.getenv("TF_CACHE_DIR", "./cache/tf")))

    # Macula settings
    macula_data_dir: Path = field(default_factory=lambda: Path(os.getenv("MACULA_DATA_DIR", "./corpora/macula")))

    # External APIs
    bible_api_key: str = field(default_factory=lambda: os.getenv("BIBLE_API_KEY", ""))
    patristic_api_url: str = field(default_factory=lambda: os.getenv("PATRISTIC_API_URL", ""))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "./logs")))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_SIZE", "10485760")))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    log_to_console: bool = field(default_factory=lambda: os.getenv("LOG_TO_CONSOLE", "true").lower() == "true")
    log_to_file: bool = field(default_factory=lambda: os.getenv("LOG_TO_FILE", "true").lower() == "true")


@dataclass
class ObservabilityConfig:
    """
    OpenTelemetry observability configuration for distributed tracing,
    metrics, and structured logging.

    Supports export to Jaeger, Tempo, Grafana, or any OTLP-compatible backend.
    """
    # Enable/disable observability
    enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_ENABLED", "true").lower() == "true"
    )

    # Service identification
    service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "biblos-v2")
    )
    service_version: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_VERSION", "2.0.0")
    )

    # OTLP endpoint configuration
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    otlp_insecure: bool = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    )

    # Tracing configuration
    tracing_enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"
    )
    sample_rate: float = field(
        default_factory=lambda: float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))
    )
    trace_console_export: bool = field(
        default_factory=lambda: os.getenv("OTEL_TRACE_CONSOLE", "false").lower() == "true"
    )

    # Metrics configuration
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true"
    )
    metrics_export_interval: int = field(
        default_factory=lambda: int(os.getenv("OTEL_METRICS_INTERVAL", "60000"))
    )

    # Logging configuration
    log_level: str = field(
        default_factory=lambda: os.getenv("OTEL_LOG_LEVEL", "INFO")
    )
    log_json_format: bool = field(
        default_factory=lambda: os.getenv("OTEL_LOG_JSON", "true").lower() == "true"
    )
    log_trace_context: bool = field(
        default_factory=lambda: os.getenv("OTEL_LOG_TRACE_CONTEXT", "true").lower() == "true"
    )

    # Resource attributes
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

    def get_sample_rate_for_env(self) -> float:
        """Get appropriate sample rate based on environment."""
        env = self.environment.lower()
        if env == "production":
            return min(self.sample_rate, 0.1)  # Max 10% in production
        elif env == "staging":
            return min(self.sample_rate, 0.5)  # Max 50% in staging
        else:
            return self.sample_rate  # Full rate in development

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for diagnostics."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "otlp_endpoint": self.otlp_endpoint,
            "tracing_enabled": self.tracing_enabled,
            "sample_rate": self.get_sample_rate_for_env(),
            "metrics_enabled": self.metrics_enabled,
            "environment": self.environment,
        }


@dataclass
class MutualTransformationConfig:
    """
    Configuration for the Mutual Transformation Metric.

    Controls thresholds and behavior for measuring bidirectional
    semantic shift between connected verses.
    """
    # Classification thresholds
    radical_threshold: float = field(
        default_factory=lambda: float(os.getenv("MTM_RADICAL_THRESHOLD", "0.4"))
    )
    moderate_threshold: float = field(
        default_factory=lambda: float(os.getenv("MTM_MODERATE_THRESHOLD", "0.2"))
    )

    # Scoring adjustments
    directionality_weight: float = field(
        default_factory=lambda: float(os.getenv("MTM_DIRECTIONALITY_WEIGHT", "0.1"))
    )

    # Feature flags
    enable_semantic_decomposition: bool = field(
        default_factory=lambda: os.getenv("MTM_SEMANTIC_DECOMPOSITION", "true").lower() == "true"
    )
    cache_embeddings: bool = field(
        default_factory=lambda: os.getenv("MTM_CACHE_EMBEDDINGS", "true").lower() == "true"
    )

    # Confidence boost for high mutual influence
    high_influence_boost: float = field(
        default_factory=lambda: float(os.getenv("MTM_HIGH_INFLUENCE_BOOST", "0.15"))
    )


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "4")))
    reload: bool = field(default_factory=lambda: os.getenv("API_RELOAD", "false").lower() == "true")
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    rate_limit: int = field(default_factory=lambda: int(os.getenv("API_RATE_LIMIT", "100")))


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    env: Environment = field(default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Data directories
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    mutual_transformation: MutualTransformationConfig = field(default_factory=MutualTransformationConfig)

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.output_dir, self.cache_dir,
                     self.ml.models_dir, self.ml.embeddings_dir,
                     self.ml.checkpoints_dir, self.logging.log_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.env == Environment.DEVELOPMENT

    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        handlers = []

        if self.logging.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(console_handler)

        if self.logging.log_to_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.log_dir / "biblos.log",
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(file_handler)

        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            handlers=handlers
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive values)."""
        return {
            "env": self.env.value,
            "debug": self.debug,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "pipeline": {
                "parallel_phases": self.pipeline.parallel_phases,
                "max_parallel_agents": self.pipeline.max_parallel_agents,
                "min_confidence": self.pipeline.min_confidence
            },
            "ml": {
                "device": self.ml.device,
                "batch_size": self.ml.batch_size,
                "embedding_model": self.ml.embedding_model
            }
        }


# Singleton configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create configuration singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global _config
    load_dotenv(override=True)
    _config = Config()
    return _config


# Phase configurations for the 4-phase SDES pipeline
PHASE_CONFIGS = {
    "linguistic": {
        "name": "linguistic",
        "agents": [
            "grammateus",
            "morphologos",
            "syntaktikos",
            "semantikos",
            "phonologos",
            "lexikos"
        ],
        "parallel": True,
        "timeout_seconds": 300,
        "dependencies": []
    },
    "theological": {
        "name": "theological",
        "agents": [
            "patrologos",
            "typologos",
            "theologos",
            "liturgikos",
            "dogmatikos"
        ],
        "parallel": True,
        "timeout_seconds": 300,
        "dependencies": ["linguistic"]
    },
    "intertextual": {
        "name": "intertextual",
        "agents": [
            "syndesmos",
            "harmonikos",
            "allographos",
            "paradeigma",
            "topos"
        ],
        "parallel": True,
        "timeout_seconds": 300,
        "dependencies": ["linguistic", "theological"]
    },
    "validation": {
        "name": "validation",
        "agents": [
            "elengchos",
            "symphonos",
            "akribeia",
            "martyres"
        ],
        "parallel": True,
        "timeout_seconds": 300,
        "dependencies": ["linguistic", "theological", "intertextual"]
    }
}


# Book codes for canonical ordering
BOOK_ORDER = [
    # Old Testament
    "GEN", "EXO", "LEV", "NUM", "DEU",
    "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR",
    "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SNG", "ISA", "JER", "LAM",
    "EZK", "DAN", "HOS", "JOL", "AMO",
    "OBA", "JON", "MIC", "NAH", "HAB",
    "ZEP", "HAG", "ZEC", "MAL",
    # New Testament
    "MAT", "MRK", "LUK", "JHN", "ACT",
    "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI",
    "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN",
    "JUD", "REV"
]


# Connection type weights for scoring
CONNECTION_WEIGHTS = {
    "typological": 1.0,
    "prophetic": 0.95,
    "verbal": 0.9,
    "thematic": 0.85,
    "conceptual": 0.8,
    "historical": 0.75,
    "liturgical": 0.7,
    "narrative": 0.65,
    "genealogical": 0.6,
    "geographical": 0.55
}
