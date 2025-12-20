"""Extractors package for Code-Orchestrator-Service."""

from src.extractors.metadata_extractor import (
    ExtractionResult,
    MetadataExtractor,
    MetadataExtractorConfig,
    get_metadata_extractor,
)

__all__ = [
    "ExtractionResult",
    "MetadataExtractor",
    "MetadataExtractorConfig",
    "get_metadata_extractor",
]
