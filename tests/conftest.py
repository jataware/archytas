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
    parser.addoption(
        "--openai-model",
        action="store",
        default="gpt-5",
        help="Comma-delimited list of OpenAI models to test (default: gpt-5)"
    )
    parser.addoption(
        "--anthropic-model",
        action="store",
        default="claude-sonnet-4-5-20250929",
        help="Comma-delimited list of Anthropic models to test (default: claude-sonnet-4-5-20250929)"
    )
    parser.addoption(
        "--gemini-model",
        action="store",
        default="gemini-2.5-pro",
        help="Comma-delimited list of Gemini models to test (default: gemini-2.5-pro)"
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests based on available model providers and models."""
    if "model_fixture" in metafunc.fixturenames:
        provider_option = metafunc.config.getoption("model_provider")

        model_configs = []

        if provider_option in ["all", "openai"] and os.environ.get("OPENAI_API_KEY"):
            openai_models = metafunc.config.getoption("--openai-model").split(",")
            for model_name in openai_models:
                model_configs.append(f"openai:{model_name.strip()}")

        if provider_option in ["all", "anthropic"] and os.environ.get("ANTHROPIC_API_KEY"):
            anthropic_models = metafunc.config.getoption("--anthropic-model").split(",")
            for model_name in anthropic_models:
                model_configs.append(f"anthropic:{model_name.strip()}")

        if provider_option in ["all", "gemini"] and os.environ.get("GEMINI_API_KEY"):
            gemini_models = metafunc.config.getoption("--gemini-model").split(",")
            for model_name in gemini_models:
                model_configs.append(f"gemini:{model_name.strip()}")

        if not model_configs:
            pytest.skip(f"No API keys found for provider: {provider_option}")

        metafunc.parametrize("model_fixture", model_configs, indirect=True)


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
def openai_model(openai_api_key, request):
    """Create an OpenAI model instance for testing."""
    model_name = request.config.getoption("--openai-model")
    return OpenAIModel({"api_key": openai_api_key, "model_name": model_name})


@pytest.fixture
def anthropic_model(anthropic_api_key, request):
    """Create an Anthropic model instance for testing."""
    model_name = request.config.getoption("--anthropic-model")
    return AnthropicModel({"api_key": anthropic_api_key, "model_name": model_name})


@pytest.fixture
def gemini_model(gemini_api_key, request):
    """Create a Gemini model instance for testing."""
    model_name = request.config.getoption("--gemini-model")
    return GeminiModel({"api_key": gemini_api_key, "model_name": model_name})


@pytest.fixture
def model_fixture(request):
    """Indirect fixture that provides the requested model based on provider:model_name format."""
    config = request.param  # Format: "provider:model_name"
    provider, model_name = config.split(":", 1)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return OpenAIModel({"api_key": api_key, "model_name": model_name})
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return AnthropicModel({"api_key": api_key, "model_name": model_name})
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not set")
        return GeminiModel({"api_key": api_key, "model_name": model_name})
    else:
        pytest.fail(f"Unknown provider: {provider}")


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
