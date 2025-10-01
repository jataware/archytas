import os
import pytest
from archytas.react import ReActAgent
from archytas.models.openai import OpenAIModel


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def openai_model(api_key):
    """Create an OpenAI model instance for testing."""
    return OpenAIModel({"api_key": api_key, "model_name": "gpt-5"})


@pytest.fixture
def react_agent(openai_model):
    """Create a basic ReActAgent with temperature=0 for deterministic outputs."""
    return ReActAgent(
        model=openai_model,
        temperature=0.0,
        verbose=True,
        allow_ask_user=False,
    )


@pytest.fixture
def react_agent_with_tools(openai_model):
    """Factory fixture to create ReActAgent with custom tools."""
    def _create_agent(tools, **kwargs):
        return ReActAgent(
            model=openai_model,
            tools=tools,
            temperature=0.0,
            verbose=True,
            allow_ask_user=False,
            **kwargs
        )
    return _create_agent
