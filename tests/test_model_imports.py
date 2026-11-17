"""The namespaced implementations and compatibility exports share identities."""
from importlib import import_module

import pytest


@pytest.mark.parametrize("module_name,model_name,provider_name", [
    ("anthropic", "AnthropicModel", "AnthropicProvider"),
    ("azure", "AzureOpenAIModel", "AzureOpenAIProvider"),
    ("bedrock", "BedrockModel", "BedrockProvider"),
    ("gemini", "GeminiModel", "GeminiProvider"),
    ("groq", "GroqModel", "GroqProvider"),
    ("ollama", "OllamaModel", "OllamaProvider"),
    ("openai", "OpenAIModel", "OpenAIProvider"),
    ("openrouter", "OpenRouterModel", "OpenRouterProvider"),
])
def test_provider_compatibility_exports(module_name, model_name, provider_name):
    wrapper = import_module(f"archytas.models.{module_name}")
    implementation = import_module(f"archytas.models.providers.{module_name}")
    provider = getattr(implementation, provider_name)
    assert getattr(wrapper, provider_name) is provider
    assert getattr(wrapper, model_name).DEFAULT_PROVIDER is provider
    assert provider.__module__ == implementation.__name__


@pytest.mark.parametrize("module_name,family_name,slug", [
    ("base", "ModelFamily", "generic"),
    ("anthropic", "AnthropicModelFamily", "anthropic"),
    ("gemini", "GeminiModelFamily", "gemini"),
    ("gemini", "Gemini2ModelFamily", "gemini2"),
    ("gemini", "Gemini3ModelFamily", "gemini3"),
    ("llama", "LlamaModelFamily", "llama"),
    ("openai", "OpenAIModelFamily", "openai"),
])
def test_family_exports_and_registry(module_name, family_name, slug):
    models = import_module("archytas.models")
    families = import_module("archytas.models.families")
    implementation = import_module(f"archytas.models.families.{module_name}")
    family = getattr(implementation, family_name)
    assert getattr(models, family_name) is family
    assert getattr(families, family_name) is family
    name, resolved = families.resolve_model_family(f"{slug}:model", families.ModelFamily())
    assert name == "model"
    assert type(resolved) is family


def test_provider_interface_compatibility_export():
    from archytas.models import ModelProvider
    from archytas.models.provider import ModelProvider as LegacyModelProvider
    from archytas.models.providers import ModelProvider as PackageModelProvider
    from archytas.models.providers.base import ModelProvider as BaseModelProvider

    assert ModelProvider is LegacyModelProvider is PackageModelProvider is BaseModelProvider
