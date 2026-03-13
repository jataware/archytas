from __future__ import annotations

from typing import ClassVar

from ..base_family import BaseModelFamily, ModelVersion


class GenericFamily(BaseModelFamily):
    """Fallback model family for unknown/unrecognized models."""

    VERSIONS: ClassVar[dict[str, ModelVersion]] = {
        "latest": ModelVersion(model_name="unknown"),
    }
    DEFAULT_VERSION: ClassVar[str] = "latest"
    PROMPT_INSTRUCTIONS: ClassVar[str] = ""

    def context_size(self, model_name: str) -> int | None:
        return None
