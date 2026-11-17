"""Model behavior interfaces and explicit, extensible family-name resolution."""
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from ...agent import AgentResponse
    from ..base import BaseArchytasModel

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


def resolve_model_family(model_name: str, default: ModelFamily | None) -> tuple[str, ModelFamily | None]:
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


register_model_family(ModelFamily)
