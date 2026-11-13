"""Model behavior and explicit, extensible family-name resolution."""
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from ..agent import AgentResponse
    from .base import BaseArchytasModel

from langchain_core.messages import AIMessage, SystemMessage


@dataclass(frozen=True)
class ReasoningSettings:
    """Family reasoning preferences, independent of API parameter names."""

    effort: str | None = None
    summary: str | None = None
    budget: int | None = None
    include_thoughts: bool = False


class ModelFamily:
    """Semantic model behavior; API-specific encoding belongs to providers.

    Subclasses can register stable versioned slugs (for example ``gemini3``)
    without changing the meaning of an existing family's slug.
    """

    slug: ClassVar[str] = "generic"
    MODEL_PROMPT_INSTRUCTIONS: ClassVar[str] = ""

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        return messages

    def rectify_result(self, model: "BaseArchytasModel", response: AIMessage) -> AIMessage:
        return response

    def process_result(self, model: "BaseArchytasModel", result: "AgentResponse") -> "AgentResponse":
        return result

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        return None

    def supported_reasoning_efforts(self, model_name: str, profile: dict | None = None) -> list[str] | None:
        return None

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> ReasoningSettings | None:
        return None

    def prepare_invoke_kwargs(self, model: "BaseArchytasModel", kwargs: dict) -> dict:
        profile = getattr(model._model, "profile", None) or {}
        if kwargs.get("temperature") is None or profile.get("temperature") is not True:
            kwargs.pop("temperature", None)
        return kwargs


_FAMILIES: dict[str, type[ModelFamily]] = {}


def register_model_family(family: type[ModelFamily]) -> type[ModelFamily]:
    """Register a family once; existing slug meanings cannot be replaced."""
    slug = family.slug
    if not slug or ":" in slug:
        raise ValueError("Model family slugs must be nonempty and cannot contain ':'.")
    if slug in _FAMILIES and _FAMILIES[slug] is not family:
        raise ValueError(f"Model family slug already registered: {slug}")
    _FAMILIES[slug] = family
    return family


def resolve_model_family(model_name: str, default: ModelFamily) -> tuple[str, ModelFamily]:
    """Strip only registered leading prefixes; leave provider IDs intact."""
    slug, separator, name = model_name.partition(":")
    family = _FAMILIES.get(slug) if separator else None
    if family is None:
        return model_name, default
    if not name:
        raise ValueError(f"Missing model name after family prefix '{slug}:'")
    return name, family()


def _combine_system_messages(messages: "list[BaseMessage]") -> "list[BaseMessage]":
    output = []
    system = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system.append(message.content)
        else:
            output.append(message)
    output.insert(0, SystemMessage(content="\n".join(system)))
    return output


def _supported_efforts(
    model_name: str, profile: dict | None, levels: dict[str, list[str]],
) -> list[str] | None:
    if profile is not None:
        if "reasoning_effort_levels" in profile:
            return profile["reasoning_effort_levels"] or None
        if profile.get("reasoning_output") is False:
            return None
    for prefix in sorted(levels, key=len, reverse=True):
        if model_name.lower().startswith(prefix):
            return levels[prefix]
    return None


@register_model_family
class OpenAIModelFamily(ModelFamily):
    slug = "openai"

    def supported_reasoning_efforts(self, model_name: str, profile: dict | None = None) -> list[str] | None:
        # Keep the existing public constants and environment settings usable.
        from .openai import REASONING_EFFORT_LEVELS
        return _supported_efforts(model_name, profile, REASONING_EFFORT_LEVELS)

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> ReasoningSettings | None:
        from .openai import PREFERRED_EFFORT_ORDER, REASONING_EFFORT_REQUEST, REASONING_SUMMARY_REQUEST
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


@register_model_family
class AnthropicModelFamily(ModelFamily):
    slug = "anthropic"

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        return _combine_system_messages(messages)

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        name = model_name or model.model_name
        return 200_000 if name and ("haiku" in name or "sonnet" in name) else None


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
        from .gemini import REASONING_EFFORT_LEVELS
        return _supported_efforts(model_name, profile, REASONING_EFFORT_LEVELS)

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> ReasoningSettings | None:
        from .gemini import (
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


@register_model_family
class LlamaModelFamily(ModelFamily):
    slug = "llama"
    MODEL_PROMPT_INSTRUCTIONS = """
If you receieve a message with a role of `tool` in response to a tool being called, the user has not seen the content
of that message. Those messages are generated by the system and sent to you for review as part of the ReAct loop. Please
treat such messages as a response from a third party that provide more information for you to use to help achieve the
user's task.
Be sure to always use the `final_answer` tool to report back to the user.
"""


class LegacyGroqModelFamily(LlamaModelFamily):
    """Preserve the old Groq wrapper policy unless a family is selected."""

    MODEL_PROMPT_INSTRUCTIONS = """\
When generating JSON, remember to not wrap strings in triple quotes such as \'\'\' or \"\"\". If you want to add newlines \
to the JSON text, use `\\n` to add newlines.
Ensure all generated JSON is valid and would pass a JSON validator.
"""

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        output = _combine_system_messages(messages)
        for index, message in enumerate(output):
            if isinstance(message, AIMessage) and message.tool_calls:
                copied = message.model_copy(deep=True)
                copied.tool_calls = []
                output[index] = copied
        return output


register_model_family(ModelFamily)
