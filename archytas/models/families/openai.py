"""OpenAI model-family behavior."""
from typing import TYPE_CHECKING

from .base import ModelFamily, ReasoningSettings, _supported_efforts, register_model_family

if TYPE_CHECKING:
    from ..base import BaseArchytasModel


@register_model_family
class OpenAIModelFamily(ModelFamily):
    slug = "openai"

    def supported_reasoning_efforts(self, model_name: str, profile: dict | None = None) -> list[str] | None:
        # Keep the existing public constants and environment settings usable.
        from ..openai import REASONING_EFFORT_LEVELS
        return _supported_efforts(model_name, profile, REASONING_EFFORT_LEVELS)

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> ReasoningSettings | None:
        from ..openai import PREFERRED_EFFORT_ORDER, REASONING_EFFORT_REQUEST, REASONING_SUMMARY_REQUEST
        supported = model._get_supported_reasoning_efforts(model_name, profile)
        if supported is None:
            return None
        candidates = [REASONING_EFFORT_REQUEST, *PREFERRED_EFFORT_ORDER]
        for effort in candidates:
            if effort in supported:
                return ReasoningSettings(effort=effort, summary=REASONING_SUMMARY_REQUEST)
        return None

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        from langchain_openai.llms.base import OpenAI
        name = model_name or model.model_name
        try:
            return OpenAI.modelname_to_contextsize(name)
        except ValueError:
            if 'gpt-4.1' in name:
                return 1_000_000
            if name.startswith(('o3', 'o4')):
                return 200_000
            if 'gpt-5' in name.lower():
                return 400_000
            return None
