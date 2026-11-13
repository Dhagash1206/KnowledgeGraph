"""Configuration — pydantic-settings v2 with YAML overlay and canonical model paths."""
from __future__ import annotations

import os
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # Pipeline
    pipeline_mode: Literal["fast", "accurate"] = "accurate"
    max_upload_mb: int = 50
    window_size: int = 3
    window_stride: int = 2
    debug_pipeline: bool = False
    intermediate_graph_dir: str = "Datasets/Processed"

    # Canonical Model & Whitelist Paths
    relation_scorer_model_path: str = "TrainedModels/RelationScorer.joblib"
    entity_resolver_model_path: str = "TrainedModels/EntityResolver.joblib"
    relations_config_path: str = "Configuration/RelationTypes.yaml"
    relation_backend: Literal["rules", "learned", "encoder", "contextual_v2", "classical_news", "semeval_svm"] = "rules"
    utd_svm_model_path: str = "TrainedModels/LexicalSvm"
    learned_relation_model_path: str = "TrainedModels/LearnedRelations.joblib"
    contextual_v2_model_path: str = "TrainedModels/ContextualV2"
    contextual_v2_allow_unverified: bool = False

    # Entity resolution
    resolution_threshold: float = 0.85
    min_related_confidence: float = 0.60

    # Auth (empty = disabled)
    api_key: str = ""
    # Browser origins allowed to call the API cross-origin (JSON list in .env).
    cors_allow_origins: list[str] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win: explicit kwargs > process env > .env > YAML > defaults.
        # Every source is validated, so a bad YAML value raises instead of being accepted.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlOverlaySource(settings_cls),
            file_secret_settings,
        )


def _resolve_yaml_path() -> str | None:
    """CONFIG_YAML env var, then Configuration/Settings.yaml, then legacy Settings.yaml."""
    path_str = os.environ.get("CONFIG_YAML")
    if path_str:
        return path_str
    for candidate in ("Configuration/Settings.yaml", "Settings.yaml"):
        if os.path.exists(candidate):
            return candidate
    return None


class YamlOverlaySource(PydanticBaseSettingsSource):
    """Non-secret settings from the YAML config file."""

    _SECRETS = frozenset({"neo4j_password", "api_key"})

    def get_field_value(self, field, field_name):  # required by the base class; unused
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path_str = _resolve_yaml_path()
        if not path_str or not os.path.exists(path_str):
            return {}
        with open(path_str, "r", encoding="utf-8") as config_file:
            overrides: dict = yaml.safe_load(config_file) or {}
        return {
            key: value
            for key, value in overrides.items()
            if key in self.settings_cls.model_fields and key not in self._SECRETS
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
