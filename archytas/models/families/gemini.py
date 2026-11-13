"""Gemini model-family behavior and generation-specific extension points."""
from typing import TYPE_CHECKING

from .base import (
    ModelFamily, ReasoningSettings, _combine_system_messages,
    _supported_efforts, register_model_family,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from ..base import BaseArchytasModel


@register_model_family
class GeminiModelFamily(ModelFamily):
    slug = "gemini"
    MODEL_PROMPT_INSTRUCTIONS = """
Before you call EACH and ANY tool, you MUST add a text block to the output a plain-text 1-3 sentence explanation of reasoning why a particular tool or code generation was selected for use. This will be shown to the user so the user can keep track of what you are up to, so be succinct, polite, and helpful.
This must be done for EVERY tool call. When preparing to use a tool, you must explicitly address the user. Your internal thinking is for you; your text output is for the user. Always write a short text message directly to the user explaining what you are about to do before the actual tool invocation.

IMPORTANT: You possess an internal thinking/reasoning process, but the user CANNOT see this as direct communication. Therefore, before generating ANY tool call, you MUST output a standard, top-level text response (e.g., a "type": "text" block) containing a 1-3 sentence conversational update for the user. Do not bury your explanation to the user inside your internal thinking block.

Remember your internal reasoning and thoughts are communicated to the user. You must communicate with the user by including a text block in your chat generation output.
"""

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        return _combine_system_messages(messages)

    def supported_reasoning_efforts(self, model_name: str, profile: dict | None = None) -> list[str] | None:
        from ..gemini import REASONING_EFFORT_LEVELS
        return _supported_efforts(model_name, profile, REASONING_EFFORT_LEVELS)

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> ReasoningSettings | None:
        from ..gemini import (
            DEFAULT_REASONING_BUDGET, PREFERRED_EFFORT_ORDER,
            REASONING_BUDGET_REQUEST, REASONING_EFFORT_REQUEST,
        )
        supported = model._get_supported_reasoning_efforts(model_name, profile)
        if supported is None:
            return None
        effort = next((item for item in [REASONING_EFFORT_REQUEST, *PREFERRED_EFFORT_ORDER] if item in supported), None)
        if effort == "budget":
            budget = DEFAULT_REASONING_BUDGET
            if REASONING_BUDGET_REQUEST is not None:
                try:
                    budget = int(REASONING_BUDGET_REQUEST)
                except ValueError:
                    pass
            return ReasoningSettings(budget=budget, include_thoughts=True)
        return ReasoningSettings(effort=effort, include_thoughts=True)

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        name = model_name or model.model_name
        if "flash" in name:
            return 1_048_576
        if "pro" in name:
            return 2_097_152 if "2." in name else 1_048_576
        return None


@register_model_family
class Gemini2ModelFamily(GeminiModelFamily):
    slug = "gemini2"


@register_model_family
class Gemini3ModelFamily(GeminiModelFamily):
    slug = "gemini3"
