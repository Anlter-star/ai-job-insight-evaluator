"""Auditable normalization for list-valued extraction labels."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.data_loader import PROJECT_ROOT


DEFAULT_ALIASES_PATH = PROJECT_ROOT / "data" / "normalization_aliases.json"


def normalize_text(value: str) -> str:
    """Normalize case, width, whitespace, and punctuation for comparison."""

    text = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


class AliasNormalizer:
    """Apply field-specific, human-reviewed aliases, including one-to-many rules."""

    def __init__(self, aliases_path: Path = DEFAULT_ALIASES_PATH):
        payload = json.loads(aliases_path.read_text(encoding="utf-8"))
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("Normalization aliases must contain a fields object")
        self.schema_version = str(payload.get("schema_version", "unknown"))
        grounding_aliases = payload.get("grounding_aliases", {})
        if not isinstance(grounding_aliases, dict):
            raise ValueError("grounding_aliases must be an object")
        self.grounding_aliases = {
            normalize_text(source): [
                target for target in ([targets] if isinstance(targets, str) else targets)
            ]
            for source, targets in grounding_aliases.items()
        }
        self.aliases: dict[str, dict[str, list[str]]] = {}
        for field, mappings in fields.items():
            if not isinstance(mappings, dict):
                raise ValueError(f"Alias mappings for {field} must be an object")
            normalized_mappings: dict[str, list[str]] = {}
            for source, targets in mappings.items():
                target_list = [targets] if isinstance(targets, str) else targets
                if not isinstance(target_list, list) or not all(
                    isinstance(item, str) for item in target_list
                ):
                    raise ValueError(f"Invalid alias target for {field}: {source}")
                normalized_mappings[normalize_text(source)] = target_list
            self.aliases[field] = normalized_mappings

    def expand(self, field: str, value: str) -> list[str]:
        """Return canonical labels for a value, or the original label when unmapped."""

        return self.aliases.get(field, {}).get(normalize_text(value), [value])

    def normalized_set(self, field: str, values: list[str]) -> set[str]:
        return {
            normalize_text(target)
            for value in values
            for target in self.expand(field, value)
            if normalize_text(target)
        }

    def alias_targets(self, field: str, value: str) -> list[str] | None:
        return self.aliases.get(field, {}).get(normalize_text(value))

    def is_supported(self, value: str, source_text: str) -> bool:
        if is_supported_by_source(value, source_text):
            return True
        source = normalize_text(source_text)
        return any(
            normalize_text(phrase) in source
            for phrase in self.grounding_aliases.get(normalize_text(value), [])
        )


def is_supported_by_source(value: str, source_text: str) -> bool:
    """Conservatively check whether a predicted phrase is grounded in the JD text."""

    source = normalize_text(source_text)
    candidate = normalize_text(value)
    if candidate and candidate in source:
        return True

    # Only remove generic framing words. Short fragments are rejected to avoid
    # declaring a prediction grounded because two or three characters overlap.
    reduced = candidate
    for token in ("具备", "熟练使用", "熟悉", "了解", "相关", "能力", "经验"):
        reduced = reduced.replace(normalize_text(token), "")
    return len(reduced) >= 4 and reduced in source
