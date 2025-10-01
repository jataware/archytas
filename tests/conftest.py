import os
import pytest
from archytas.react import ReActAgent
from archytas.models.openai import OpenAIModel
from archytas.models.anthropic import AnthropicModel
from archytas.models.gemini import GeminiModel


def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--model-provider",
        action="store",
        default="all",
        help="Model provider to test: openai, anthropic, gemini, or all"
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests based on available model providers."""
    if "model_fixture" in metafunc.fixturenames:
        provider_option = metafunc.config.getoption("model_provider")

        available_providers = []

        if provider_option in ["all", "openai"] and os.environ.get("OPENAI_API_KEY"):
            available_providers.append("openai_model")

        if provider_option in ["all", "anthropic"] and os.environ.get("ANTHROPIC_API_KEY"):
            available_providers.append("anthropic_model")

        if provider_option in ["all", "gemini"] and os.environ.get("GEMINI_API_KEY"):
            available_providers.append("gemini_model")

        if not available_providers:
            pytest.skip(f"No API keys found for provider: {provider_option}")

        metafunc.parametrize("model_fixture", available_providers, indirect=True)


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def gemini_api_key():
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture
def openai_model(openai_api_key):
    """Create an OpenAI model instance for testing."""
    return OpenAIModel({"api_key": openai_api_key, "model_name": "gpt-5"})


@pytest.fixture
def anthropic_model(anthropic_api_key):
    """Create an Anthropic model instance for testing."""
    return AnthropicModel({"api_key": anthropic_api_key, "model_name": "claude-sonnet-4-5-20250929"})


@pytest.fixture
def gemini_model(gemini_api_key):
    """Create a Gemini model instance for testing."""
    return GeminiModel({"api_key": gemini_api_key, "model_name": "gemini-2.5-flash"})


@pytest.fixture
def model_fixture(request):
    """Indirect fixture that provides the requested model."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def react_agent(model_fixture):
    """Create a basic ReActAgent with temperature=0 for deterministic outputs."""
    return ReActAgent(
        model=model_fixture,
        temperature=0.0,
        verbose=True,
        allow_ask_user=False,
    )


@pytest.fixture
def react_agent_with_tools(model_fixture):
    """Factory fixture to create ReActAgent with custom tools."""
    def _create_agent(tools, **kwargs):
        return ReActAgent(
            model=model_fixture,
            tools=tools,
            temperature=0.0,
            verbose=True,
            allow_ask_user=False,
            **kwargs
        )
    return _create_agent
