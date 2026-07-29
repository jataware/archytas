"""BaseArchytasModel.invoke must not forward agent_tools to the provider.

agent_tools is an archytas-level argument. LangChain passes kwargs it does not
recognize straight into the provider request, so forwarding it reaches the
OpenAI client as an unexpected keyword and raises TypeError before any request
is made. It was forwarded unconditionally, including as None, so every sync
caller hit it regardless of whether tools were passed (for example
beaker-notebook's streamline exporter, used by its AI export). ainvoke has
always consumed it.
"""
from types import SimpleNamespace

import pytest

from archytas.models.base import BaseArchytasModel, ModelConfig


class StrictChatModel:
    """Chat model that rejects unknown kwargs, as the OpenAI client does."""

    def __init__(self):
        self.bound_tools = None
        self.invoke_kwargs = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, input=None, config=None, *, stop=None):
        self.invoke_kwargs = {"stop": stop}
        return SimpleNamespace(content="ok")


class StrictModel(BaseArchytasModel):
    DEFAULT_MODEL = "strict-fake"

    def __init__(self, **kwargs):
        super().__init__(ModelConfig(model_name="strict-fake"), **kwargs)

    def initialize_model(self, **kwargs):
        return StrictChatModel()


@pytest.mark.unit
def test_invoke_does_not_forward_agent_tools():
    model = StrictModel()
    # An empty tool dict is enough: it is not None, so it takes the same branch
    # a real tool set does, without needing tool objects to convert.
    result = model.invoke(["hello"], agent_tools={})
    assert result.content == "ok"
    assert model._model.invoke_kwargs == {"stop": None}


@pytest.mark.unit
def test_invoke_binds_tools_like_ainvoke():
    model = StrictModel()
    assert model.lc_tools is None
    model.invoke(["hello"], agent_tools={})
    assert model.lc_tools is not None
    assert model._model.bound_tools is model.lc_tools


@pytest.mark.unit
def test_invoke_without_agent_tools_leaves_tools_unbound():
    model = StrictModel()
    model.invoke(["hello"])
    assert model.lc_tools is None
    assert model._model.bound_tools is None
