"""Regression test for Agent.set_model keeping the chat history in sync.

The chat history holds its own model reference (used for token budgeting and
the model metadata it serializes for UIs). Assigning ``agent.model`` alone
leaves that reference stale; ``Agent.set_model`` must update both.
"""
from archytas.agent import Agent
from archytas.models.openai import OpenAIModel


def test_set_model_updates_agent_and_chat_history(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    first = OpenAIModel({"api_key": "sk-a", "model_name": "gpt-4o-mini"})
    second = OpenAIModel({"api_key": "sk-b", "model_name": "gpt-4o"})

    agent = Agent(model=first, spinner=None, rich_print=False)
    assert agent.model is first
    assert agent.chat_history.model is first

    agent.set_model(second)
    assert agent.model is second
    assert agent.chat_history.model is second
