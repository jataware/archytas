"""Offline checks for the opt-in Bedrock live-test configuration."""
from types import SimpleNamespace

import pytest

from tests import conftest


class FakeMetafunc:
    fixturenames = ["model_fixture"]

    def __init__(self, provider, models):
        self.config = SimpleNamespace(getoption=lambda option: {
            "model_provider": provider,
            "--bedrock-model": models,
        }[option])
        self.parameters = None

    def parametrize(self, name, values, indirect):
        assert name == "model_fixture"
        assert indirect is True
        self.parameters = values


@pytest.mark.parametrize("provider", ["all", "bedrock"])
def test_bedrock_live_tests_require_opt_in(monkeypatch, provider):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARCHYTAS_BEDROCK_TESTS_ENABLED", "0")
    metafunc = FakeMetafunc(provider, "openai:model")
    with pytest.raises(pytest.skip.Exception):
        conftest.pytest_generate_tests(metafunc)


@pytest.mark.parametrize("provider", ["all", "bedrock"])
def test_bedrock_live_tests_parameterize_families(monkeypatch, provider):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARCHYTAS_BEDROCK_TESTS_ENABLED", "1")
    metafunc = FakeMetafunc(provider, "openai:openai-id, anthropic:claude-id:0")
    conftest.pytest_generate_tests(metafunc)
    assert metafunc.parameters == ["bedrock:openai:openai-id", "bedrock:anthropic:claude-id:0"]


def test_bedrock_fixture_preserves_family_and_model_id(monkeypatch):
    monkeypatch.setattr(conftest, "BedrockModel", lambda config: config)
    request = SimpleNamespace(param="bedrock:anthropic:claude-id:0")
    assert conftest.model_fixture.__wrapped__(request) == {"model_name": "anthropic:claude-id:0"}
