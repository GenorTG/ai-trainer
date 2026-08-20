"""Configuration loader for the inference server.

WHAT THIS FILE DOES
==================
Loads settings from a YAML file and gives the rest of the code easy
access to them. Think of it as a "settings object" that lives in a file.

KEY CONCEPTS
============
- YAML: a human-readable data format (like JSON but with less punctuation).
- Pydantic: a Python library for data validation. Used here to ensure
  the config has the right types and structure.
- Default config: if no file is provided, we use sensible defaults.
- Environment overrides: env vars can override config file values
  (useful for production deployments).
"""

"""Configuration — YAML-based config for portable server."""
import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False

@dataclass
class ModelConfig:
    path: str = ""
    n_gpu_layers: int = 99
    n_ctx: int = 8192
    n_threads: int = 4
    verbose: bool = False

@dataclass
class RAGConfig:
    enabled: bool = True
    store_path: str = "rag_data/store"
    documents_path: str = "rag_data/documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    min_score: float = 0.3
    max_context_length: int = 2000

@dataclass
class InferenceConfig:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05
    min_p: float = 0.05

@dataclass
class APIConfig:
    key: str = ""
    cors: bool = True
    rate_limit: int = 100  # requests per minute

@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: APIConfig = field(default_factory=APIConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load config from YAML file, with defaults for missing fields."""
    config = AppConfig()

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            if "server" in data:
                for k, v in data["server"].items():
                    if hasattr(config.server, k):
                        setattr(config.server, k, v)

            if "model" in data:
                for k, v in data["model"].items():
                    if hasattr(config.model, k):
                        setattr(config.model, k, v)

            if "rag" in data:
                for k, v in data["rag"].items():
                    if hasattr(config.rag, k):
                        setattr(config.rag, k, v)

            if "inference" in data:
                for k, v in data["inference"].items():
                    if hasattr(config.inference, k):
                        setattr(config.inference, k, v)

            if "api" in data:
                for k, v in data["api"].items():
                    if hasattr(config.api, k):
                        setattr(config.api, k, v)

        except ImportError:
            print("Warning: PyYAML not installed, using defaults")

    # Environment variable overrides
    if os.environ.get("MODEL_PATH"):
        config.model.path = os.environ["MODEL_PATH"]
    if os.environ.get("RAG_STORE_PATH"):
        config.rag.store_path = os.environ["RAG_STORE_PATH"]
    if os.environ.get("API_KEY"):
        config.api.key = os.environ["API_KEY"]
    if os.environ.get("PORT"):
        config.server.port = int(os.environ["PORT"])

    return config


def save_config(config: AppConfig, config_path: str = "config.yaml"):
    """Save config to YAML file."""
    import yaml

    data = {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
        },
        "model": {
            "path": config.model.path,
            "n_gpu_layers": config.model.n_gpu_layers,
            "n_ctx": config.model.n_ctx,
        },
        "rag": {
            "enabled": config.rag.enabled,
            "store_path": config.rag.store_path,
            "documents_path": config.rag.documents_path,
            "embedding_model": config.rag.embedding_model,
            "chunk_size": config.rag.chunk_size,
            "top_k": config.rag.top_k,
        },
        "inference": {
            "max_tokens": config.inference.max_tokens,
            "temperature": config.inference.temperature,
            "top_p": config.inference.top_p,
            "repeat_penalty": config.inference.repeat_penalty,
        },
        "api": {
            "key": config.api.key,
            "cors": config.api.cors,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
