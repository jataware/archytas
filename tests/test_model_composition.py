"""Offline coverage for independent providers and model families."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from archytas.models.base import BaseArchytasModel, ModelConfig
from archytas.models.bedrock import BedrockModel, BedrockProvider
from archytas.models.families import (
    AnthropicModelFamily, Gemini2ModelFamily, Gemini3ModelFamily,
    LlamaModelFamily, ModelFamily, OpenAIModelFamily, register_model_family,
    resolve_model_family,
)
from archytas.models.openai import OpenAIModel, OpenAIProvider
from archytas.models.openrouter import OpenRouterModel
from archytas.models.provider import ModelProvider


class FakeChatModel:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.model = kwargs.get("model", "fake")
        self.profile = {"temperature": True}
        self.openai_api_key = "dummy"
        self.calls = []

    def invoke(self, messages, config=None, stop=None, **kwargs):
        self.calls.append((messages, kwargs))
        return AIMessage(content="ok")

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        return self.invoke(messages, config, stop, **kwargs)


class FakeProvider(ModelProvider):
    def initialize_model(self, model, **kwargs):
        return FakeChatModel(model=model.config.model_name)


@pytest.mark.parametrize("name,family,wire_name", [
    ("openai:gpt-6.1-terra", OpenAIModelFamily, "gpt-6.1-terra"),
    ("anthropic:us.anthropic.claude-opus-5-5", AnthropicModelFamily, "us.anthropic.claude-opus-5-5"),
    ("gemini2:gemini-2.5-flash", Gemini2ModelFamily, "gemini-2.5-flash"),
    ("gemini3:gemini-3-flash", Gemini3ModelFamily, "gemini-3-flash"),
    ("llama:local-model:latest", LlamaModelFamily, "local-model:latest"),
])
def test_registered_prefixes(name, family, wire_name):
    model = BaseArchytasModel({"model_name": name}, provider=FakeProvider)
    assert type(model.model_family) is family
    assert model.model_name == wire_name
    assert model.config.model_name == wire_name
    assert model._model.options["model"] == wire_name


@pytest.mark.parametrize("name", [
    "local-model:latest", "us.anthropic.claude-sonnet:0",
    "arn:aws:bedrock:us-east-1:123:inference-profile/model:0",
    "unknown:model", "openai/gpt-4o", "my-openai:model",
])
def test_unknown_prefixes_preserve_provider_ids(name):
    model = BaseArchytasModel({"model_name": name}, provider=FakeProvider, model_family=LlamaModelFamily)
    assert model.model_name == name
    assert type(model.model_family) is LlamaModelFamily


def test_prefix_does_not_mutate_callers_config():
    config = ModelConfig(model_name="anthropic:model")
    model = BaseArchytasModel(config, provider=FakeProvider)
    assert config.model_name == "anthropic:model"
    assert model.config.model_name == "model"


def test_prefix_is_more_specific_than_constructor_family():
    model = BaseArchytasModel(
        {"model_name": "anthropic:model"}, provider=FakeProvider(), model_family=OpenAIModelFamily(),
    )
    assert type(model.model_family) is AnthropicModelFamily


def test_registered_empty_name_is_rejected():
    with pytest.raises(ValueError, match="Missing model name"):
        resolve_model_family("openai:", ModelFamily())


def test_versioned_family_can_override_behavior_without_changing_original():
    @register_model_family
    class OpenAIFutureFamily(OpenAIModelFamily):
        slug = "test-openai-future"
        MODEL_PROMPT_INSTRUCTIONS = "Future model instructions."

        def prepare_invoke_kwargs(self, model, kwargs):
            kwargs.pop("temperature", None)
            return kwargs

    original = BaseArchytasModel({"model_name": "openai:model"}, provider=FakeProvider)
    future = BaseArchytasModel({"model_name": "test-openai-future:model"}, provider=FakeProvider)
    assert original._prepare_invoke_kwargs({"temperature": 0.5}) == {"temperature": 0.5}
    assert future._prepare_invoke_kwargs({"temperature": 0.5}) == {}
    assert future.MODEL_PROMPT_INSTRUCTIONS == "Future model instructions."
    assert original.MODEL_PROMPT_INSTRUCTIONS == ""
    with pytest.raises(ValueError, match="already registered"):
        register_model_family(type("Duplicate", (ModelFamily,), {"slug": "test-openai-future"}))


@pytest.mark.parametrize("family,system_count", [("openai", 2), ("anthropic", 1)])
@pytest.mark.parametrize("wrapper,module", [
    (BedrockModel, "archytas.models.bedrock"),
    (OpenRouterModel, "archytas.models.openrouter"),
])
async def test_same_families_through_bedrock_and_openrouter(monkeypatch, family, system_count, wrapper, module):
    client_symbol = "ChatBedrockConverse" if wrapper is BedrockModel else "ChatOpenAI"
    monkeypatch.setattr(f"{module}.{client_symbol}", FakeChatModel)
    monkeypatch.setenv("ARCHYTAS_DISABLE_PROMPT_CACHE", "1")
    model = wrapper({"model_name": f"{family}:remote-model:0", "api_key": "dummy"})
    messages = [SystemMessage(content="first"), HumanMessage(content="hello"), SystemMessage(content="second")]
    for result in (model.invoke(messages), await model.ainvoke(messages)):
        assert result.content == "ok"
    assert model._model.options["model"] == "remote-model:0"
    for sent, _ in model._model.calls:
        assert sum(isinstance(message, SystemMessage) for message in sent) == system_count
    assert messages[0].content == "first"
    assert messages[2].content == "second"


def test_bedrock_defaults_and_family_only_cache_markers(monkeypatch):
    monkeypatch.setattr("archytas.models.bedrock.ChatBedrockConverse", FakeChatModel)
    monkeypatch.delenv("ARCHYTAS_DISABLE_PROMPT_CACHE", raising=False)
    default = BedrockModel({"model_name": ""})
    assert default.model_name == "us.openai.gpt-6.1-terra"
    assert type(default.model_family) is OpenAIModelFamily
    claude = BedrockModel({"model_name": "anthropic:us.anthropic.claude-sonnet:0"})
    claude.tail_injection_count = 1
    messages = [SystemMessage(content="system"), HumanMessage(content="persisted"), HumanMessage(content="tail")]
    assert default._prepare_messages(messages) == messages
    prepared = claude._prepare_messages(messages)
    assert prepared[1].content[0]["cache_control"] == {"type": "ephemeral"}
    assert prepared[2].content == "tail"
    assert messages[1].content == "persisted"
    assert claude.contextsize() == 200_000


def test_provider_wire_result_processing_is_independent_of_family(monkeypatch):
    monkeypatch.setattr("archytas.models.openai.ChatOpenAI", FakeChatModel)
    model = OpenAIModel({"model_name": "anthropic:claude", "api_key": "dummy"})
    response = AIMessage(content=[
        {"type": "reasoning", "summary": [{"text": "summary"}]},
        {"type": "text", "text": "answer"},
    ])
    result = model.process_result(response)
    assert result.text == "answer"
    assert result.metadata == {"reasoning": "summary"}
    assert type(model.provider) is OpenAIProvider
    assert type(model.model_family) is AnthropicModelFamily


def test_explicit_provider_override_retains_facade(monkeypatch):
    monkeypatch.setattr("archytas.models.bedrock.ChatBedrockConverse", FakeChatModel)
    model = OpenAIModel({"model_name": "anthropic:claude"}, provider=BedrockProvider)
    assert isinstance(model, BaseArchytasModel)
    assert type(model.provider) is BedrockProvider
    assert type(model.model_family) is AnthropicModelFamily
    assert model.MODEL_SPECIFIC_CONTENT_TYPES == frozenset({"reasoning_content"})


def test_content_identity_uses_composition_not_wrapper_class(monkeypatch):
    monkeypatch.setattr("archytas.models.bedrock.ChatBedrockConverse", FakeChatModel)
    first = BedrockModel({"model_name": "anthropic:model"})
    same = OpenAIModel({"model_name": "anthropic:model"}, provider=BedrockProvider)
    other_family = BedrockModel({"model_name": "openai:model"})
    other_provider = BaseArchytasModel({"model_name": "anthropic:model"}, provider=FakeProvider)
    message = first.mark_model_specific_content(AIMessage(content=[
        {"type": "reasoning_content", "signature": b"signature"},
        {"type": "text", "text": "answer"},
    ]))
    assert first.filter_model_specific_content(message, same) is message
    for model in (other_family, other_provider):
        assert first.filter_model_specific_content(message, model).content == [{"type": "text", "text": "answer"}]
    assert len(message.content) == 2


def test_legacy_subclass_hooks_still_run():
    class CustomModel(BaseArchytasModel):
        MODEL_PROMPT_INSTRUCTIONS = "Custom instructions."

        def auth(self, **kwargs):
            self.auth_called = True

        def initialize_model(self, **kwargs):
            return FakeChatModel()

        def _preprocess_messages(self, messages):
            return [HumanMessage(content="custom")]

    model = CustomModel({"model_name": "test"})
    assert model.auth_called
    assert model.MODEL_PROMPT_INSTRUCTIONS == "Custom instructions."
    assert model.invoke([]).content == "ok"
    assert model._model.calls[0][0][0].content == "custom"


def test_reasoning_preferences_are_encoded_by_provider():
    from archytas.models import ReasoningSettings
    from archytas.models.gemini import GeminiProvider

    class ReasoningFamily(ModelFamily):
        def reasoning_config(self, model, model_name, profile=None):
            return ReasoningSettings(effort="low", summary="concise", include_thoughts=True)

    model = BaseArchytasModel(
        {"model_name": "opaque-deployment"}, provider=FakeProvider, model_family=ReasoningFamily,
    )
    assert OpenAIProvider().reasoning_config(model, model.model_name) == {
        "effort": "low", "summary": "concise",
    }
    assert GeminiProvider().reasoning_config(model, model.model_name) == {
        "include_thoughts": True, "thinking_level": "low",
    }
    # Unsupported transports do not blindly send another API's options.
    assert BedrockProvider().reasoning_config(model, model.model_name) is None


def test_legacy_reasoning_capability_hook_is_preserved(monkeypatch):
    monkeypatch.setattr("archytas.models.openai.ChatOpenAI", FakeChatModel)
    monkeypatch.setattr("archytas.models.openai.REASONING_EFFORT_REQUEST", None)

    class CustomOpenAIModel(OpenAIModel):
        def _get_supported_reasoning_efforts(self, model_name, profile=None):
            return ["low"]

    model = CustomOpenAIModel({"model_name": "opaque-deployment", "api_key": "dummy"})
    assert model._model.options["reasoning"]["effort"] == "low"


def test_context_metadata_precedes_family_and_conservative_fallback():
    model = BaseArchytasModel({"model_name": "anthropic:claude-sonnet"}, provider=FakeProvider)
    model._model.profile["max_input_tokens"] = 1234
    assert model.contextsize() == 1234
    assert model.contextsize("claude-sonnet-other") == 200_000
