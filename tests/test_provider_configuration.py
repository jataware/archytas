"""Offline regression coverage for provider authentication and API settings."""
import os
from types import SimpleNamespace

import pytest

from archytas.exceptions import AuthenticationError
from archytas.models.anthropic import AnthropicModel
from archytas.models.azure import AzureOpenAIModel
from archytas.models.bedrock import BedrockModel
from archytas.models.gemini import GeminiModel
from archytas.models.groq import GroqModel


class FakeChatModel:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.model = kwargs["model"]
        self.profile = {}
        self.client = SimpleNamespace(_client=object())


PROVIDERS = [
    (AnthropicModel, "ANTHROPIC_API_KEY", "archytas.models.providers.anthropic.ChatAnthropic", AuthenticationError),
    (GeminiModel, "GEMINI_API_KEY", "archytas.models.providers.gemini.ChatGoogleGenerativeAI", AuthenticationError),
    (GroqModel, "GROQ_API_KEY", "archytas.models.providers.groq.ChatGroq", ValueError),
]


@pytest.mark.parametrize("model_class,env_name,client_symbol,error_class", PROVIDERS)
@pytest.mark.parametrize("source", ["environment", "config", "keyword"])
def test_api_key_precedence(monkeypatch, model_class, env_name, client_symbol, error_class, source):
    monkeypatch.setattr(client_symbol, FakeChatModel)
    monkeypatch.setenv(env_name, "environment-key")
    config = {"model_name": "opaque-model"}
    kwargs = {}
    if source in ("config", "keyword"):
        config["api_key"] = "config-key"
    if source == "keyword":
        kwargs["api_key"] = "keyword-key"
    model = model_class(config, **kwargs)
    expected = {"environment": "environment-key", "config": "config-key", "keyword": "keyword-key"}[source]
    assert model.api_key == expected
    assert model._model.options["api_key"] == expected
    # These providers must not write configured credentials to the environment.
    assert os.environ[env_name] == "environment-key"


@pytest.mark.parametrize("model_class,env_name,client_symbol,error_class", PROVIDERS)
def test_missing_api_key_still_raises(monkeypatch, model_class, env_name, client_symbol, error_class):
    monkeypatch.setattr(client_symbol, FakeChatModel)
    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(error_class):
        model_class({"model_name": "opaque-model"})


@pytest.fixture
def bedrock_environment(monkeypatch):
    monkeypatch.setattr("archytas.models.providers.bedrock.ChatBedrockConverse", FakeChatModel)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-session")


@pytest.mark.parametrize("source", ["environment", "config", "keyword"])
@pytest.mark.parametrize("session", [None, "configured-session"])
def test_bedrock_credentials_precedence(bedrock_environment, source, session):
    credentials = {"aws_access_key": "config-access", "aws_secret_key": "config-secret"}
    if session is not None:
        credentials["aws_session_token"] = session
    config = {"model_name": "opaque-model"}
    kwargs = {}
    if source in ("config", "keyword"):
        config.update(credentials)
    if source == "keyword":
        kwargs = {**credentials, "aws_access_key": "kw-access", "aws_secret_key": "kw-secret"}
    model = BedrockModel(config, **kwargs)
    options = model._model.options
    expected_access = {"environment": "env-access", "config": "config-access", "keyword": "kw-access"}[source]
    expected_secret = {"environment": "env-secret", "config": "config-secret", "keyword": "kw-secret"}[source]
    assert options["aws_access_key_id"] == expected_access
    assert options["aws_secret_access_key"] == expected_secret
    assert options["aws_session_token"] == ("env-session" if source == "environment" else session)


@pytest.mark.parametrize("credentials", [
    {"aws_access_key": "access"},
    {"aws_secret_key": "secret"},
    {"aws_session_token": "session"},
])
def test_bedrock_partial_keyword_credentials_are_rejected(bedrock_environment, credentials):
    with pytest.raises(AuthenticationError, match="Missing key"):
        BedrockModel({"model_name": "opaque-model"}, **credentials)


def test_bedrock_profile_precedes_environment_credentials(bedrock_environment):
    model = BedrockModel({"model_name": "opaque-model"}, credentials_profile_name="test-profile")
    assert model._model.options["credentials_profile_name"] == "test-profile"
    assert "aws_access_key_id" not in model._model.options


@pytest.mark.parametrize("config_version,keyword_version,expected", [
    (None, None, "2024-10-21"),
    ("config-version", None, "config-version"),
    (None, "keyword-version", "keyword-version"),
    ("config-version", "keyword-version", "keyword-version"),
])
def test_azure_api_version_override(monkeypatch, config_version, keyword_version, expected):
    monkeypatch.setattr("archytas.models.providers.azure.AzureChatOpenAI", FakeChatModel)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "environment-key")
    config = {"model_name": "deployment"}
    kwargs = {}
    if config_version is not None:
        config["api_version"] = config_version
    if keyword_version is not None:
        kwargs["api_version"] = keyword_version
    model = AzureOpenAIModel(config, **kwargs)
    assert model._model.options["api_version"] == expected
