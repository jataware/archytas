"""
Regression tests for PEP 563 / `from __future__ import annotations` compatibility.

When a module opts into future annotations, all function annotations become
strings at runtime. Archytas reads annotations via `inspect.signature(...)`
for both @tool dependency-injection and @statetool conditions. Prior to the
fix in `tool_utils.get_tool_signature` and `Agent._evaluate_statetool_condition`,
stringified annotations could not be matched against `INJECTION_MAPPING`
(which is keyed by the actual type objects) and DI silently no-op'd —
for @statetool conditions this meant the framework silently skipped the
injection; for @tool DI it meant the injected argument was never provided.

This file has `from __future__ import annotations` at the top so the
annotations on the test tools/conditions below are stringified at runtime,
exercising the resolver path.
"""
from __future__ import annotations

import pytest

from archytas.agent import STATE_INJECTION_PLACEHOLDER_TEXT
from archytas.tool_utils import AgentRef, statetool, tool
from archytas.react import ReActAgent


# ---- @statetool condition with stringified AgentRef -------------------------

def _make_future_annotated_statetool():
    """Condition and body both use stringified `AgentRef` annotations."""

    def fires_when_agent_present(agent: AgentRef) -> bool:
        # Under `from __future__ import annotations`, the `agent: AgentRef`
        # annotation is a string at runtime. If archytas doesn't resolve it,
        # DI fails, the call raises TypeError, the statetool is silently
        # skipped, and this condition effectively never runs.
        return agent is not None

    @statetool(condition=fires_when_agent_present, name="future_annotated_state")
    def tool_body(agent: AgentRef) -> str:
        """
        ** INTERNAL ** Return the agent's class name.

        Returns:
            str: A marker string identifying the current agent.
        """
        return f"FUTURE_ANNOTATIONS_MARKER__{type(agent).__name__}"

    return tool_body


class TestStateToolFutureAnnotations:

    @pytest.mark.asyncio
    async def test_statetool_condition_with_stringified_annotation(self, model_fixture):
        """Condition accepting `agent: AgentRef` (stringified) still fires correctly."""
        from langchain_core.messages import AIMessage

        tool_fn = _make_future_annotated_statetool()
        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        await agent.react_async("Say hello.")
        sent = agent.chat_history.last_sent_messages or []

        # Framework-fabricated placeholder AIMessage referencing our statetool.
        fabricated = [
            m for m in sent
            if isinstance(m, AIMessage)
            and m.content == STATE_INJECTION_PLACEHOLDER_TEXT
            and any(tc.get("name") == "future_annotated_state" for tc in (m.tool_calls or []))
        ]
        assert fabricated, (
            "framework injection for future-annotated statetool must fire — "
            "if this fails, the AgentRef annotation didn't resolve"
        )


# ---- @tool DI with stringified AgentRef -------------------------------------

class TestToolDIFutureAnnotations:

    def test_tool_signature_resolves_stringified_injection(self):
        """@tool with a stringified `AgentRef` annotation records it as an injection."""

        @tool
        def example_tool(query: str, agent: AgentRef) -> str:
            """
            Dummy tool for resolver testing.

            Args:
                query (str): A string to echo.

            Returns:
                str: The agent's class name combined with the query.
            """
            return f"{type(agent).__name__}:{query}"

        # _injections is populated by get_tool_signature; it must contain the
        # AgentRef entry despite the annotation being stringified.
        injections = getattr(example_tool, "_injections", {})
        assert "agent" in injections, (
            f"agent injection was not detected; _injections={injections!r}"
        )
        assert injections["agent"] is AgentRef, (
            f"agent injection resolved to the wrong type: {injections['agent']!r}"
        )

        # And the plain `query: str` arg should NOT be treated as an injection.
        args_list = getattr(example_tool, "_args_list", [])
        arg_names = [a[0] for a in args_list]
        assert "query" in arg_names
        assert "agent" not in arg_names, (
            "agent should be in _injections, not _args_list"
        )
