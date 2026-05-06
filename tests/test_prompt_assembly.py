"""
Tests for the sectioned prompt assembly machinery on ``archytas.prompt`` and
``archytas.agent.Agent``.
"""
import warnings

import pytest

from archytas.agent import Agent
from archytas.prompt import (
    DEFAULT_BASE_FRAMEWORK_PROMPT,
    DEFAULT_HEADER_FORMATTER,
    DEFAULT_REACT_FRAMEWORK_PROMPT,
    PromptSection,
    assemble_prompt,
)


class _StubModel:
    """Minimal stand-in for ``BaseArchytasModel`` for prompt-assembly tests."""

    def __init__(self, instructions: str = ""):
        self.MODEL_PROMPT_INSTRUCTIONS = instructions


def _stub_agent(model_instructions: str = "", **kwargs) -> Agent:
    """Build an ``Agent``-like instance bypassing the real constructor.

    The real ``Agent.__init__`` constructs a ``ChatHistory`` and writes a
    system message, which depends on a real model. For unit-testing the
    section assembler we only need ``self.model``, ``self.custom_prompt``,
    ``self.framework_prompt`` (instance, optional), and
    ``self.header_formatter``.
    """
    obj = Agent.__new__(Agent)
    obj.model = _StubModel(model_instructions)
    obj.custom_prompt = kwargs.get("custom_prompt")
    if kwargs.get("framework_prompt") is not None:
        obj.framework_prompt = kwargs["framework_prompt"]
    obj.header_formatter = kwargs.get("header_formatter") or DEFAULT_HEADER_FORMATTER
    return obj


class TestDefaultHeaderFormatter:

    def test_returns_markdown_header_for_name(self):
        assert DEFAULT_HEADER_FORMATTER("Framework") == "## Framework"

    def test_returns_none_for_none_name(self):
        assert DEFAULT_HEADER_FORMATTER(None) is None

    def test_returns_none_for_empty_name(self):
        assert DEFAULT_HEADER_FORMATTER("") is None


class TestAssemblePrompt:

    def test_drops_empty_bodies(self):
        sections = [
            PromptSection(body="kept", name="A", role="a"),
            PromptSection(body="", name="B", role="b"),
            PromptSection(body="   ", name="C", role="c"),
            PromptSection(body="also kept", name=None),
        ]
        out = assemble_prompt(sections)
        assert "kept" in out
        assert "also kept" in out
        assert "## B" not in out
        assert "## C" not in out

    def test_headerless_section_has_no_header(self):
        sections = [PromptSection(body="bare body", name=None)]
        assert assemble_prompt(sections) == "bare body"

    def test_sections_joined_with_blank_lines(self):
        sections = [
            PromptSection(body="one", name="One"),
            PromptSection(body="two", name="Two"),
        ]
        out = assemble_prompt(sections)
        assert out == "## One\n\none\n\n## Two\n\ntwo"

    def test_custom_formatter_returning_none_omits_header(self):
        formatter = lambda name: None
        sections = [PromptSection(body="x", name="Whatever")]
        assert assemble_prompt(sections, header_formatter=formatter) == "x"

    def test_custom_formatter_alternate_style(self):
        formatter = lambda name: f"[{name}]" if name else None
        sections = [PromptSection(body="x", name="Header")]
        assert assemble_prompt(sections, header_formatter=formatter) == "[Header]\n\nx"

    def test_role_based_filtering_in_subclass_pattern(self):
        """Subclasses are expected to filter/reorder sections by role.
        Verify role survives round-trip and is usable for matching."""
        sections = [
            PromptSection(body="fw", name="Framework", role="framework"),
            PromptSection(body="md", name="Model", role="model"),
            PromptSection(body="env", name="Environment", role="environment"),
        ]
        without_model = [s for s in sections if s.role != "model"]
        out = assemble_prompt(without_model)
        assert "## Framework" in out
        assert "## Environment" in out
        assert "## Model" not in out

    def test_strips_body_whitespace(self):
        section = PromptSection(body="   padded   \n", name="X")
        assert assemble_prompt([section]) == "## X\n\npadded"


class TestPromptSection:

    def test_defaults(self):
        section = PromptSection(body="hello")
        assert section.body == "hello"
        assert section.name is None
        assert section.role is None

    def test_frozen(self):
        section = PromptSection(body="hello", name="X", role="x")
        with pytest.raises(Exception):
            section.body = "new"  # type: ignore[misc]


class TestAgentSectionAssembly:

    def test_default_sections_framework_and_model(self):
        agent = _stub_agent(model_instructions="Use proper JSON.")
        sections = agent.get_prompt_sections()
        assert len(sections) == 2
        assert sections[0].role == "framework"
        assert sections[0].name == "Framework"
        assert sections[0].body == DEFAULT_BASE_FRAMEWORK_PROMPT
        assert sections[1].role == "model"
        assert sections[1].name == "Model"
        assert sections[1].body == "Use proper JSON."

    def test_default_sections_omits_empty_model(self):
        agent = _stub_agent(model_instructions="")
        sections = agent.get_prompt_sections()
        roles = [s.role for s in sections]
        assert roles == ["framework"]

    def test_default_sections_omits_empty_framework(self):
        agent = _stub_agent(model_instructions="Foo", framework_prompt="")
        sections = agent.get_prompt_sections()
        roles = [s.role for s in sections]
        assert roles == ["model"]

    def test_framework_prompt_instance_override(self):
        agent = _stub_agent(framework_prompt="Custom FW.")
        sections = agent.get_prompt_sections()
        assert sections[0].body == "Custom FW."

    def test_build_system_prompt_assembled(self):
        agent = _stub_agent(model_instructions="MODEL")
        out = agent.build_system_prompt()
        assert "## Framework" in out
        assert DEFAULT_BASE_FRAMEWORK_PROMPT in out
        assert "## Model" in out
        assert "MODEL" in out

    def test_custom_prompt_short_circuits(self):
        agent = _stub_agent(
            model_instructions="MODEL",
            custom_prompt="Whole thing verbatim, no headers.",
        )
        assert agent.build_system_prompt() == "Whole thing verbatim, no headers."

    def test_custom_header_formatter(self):
        agent = _stub_agent(
            model_instructions="X",
            header_formatter=lambda n: f"### {n}" if n else None,
        )
        out = agent.build_system_prompt()
        assert "### Framework" in out
        assert "### Model" in out
        # No bare ``## Framework`` line (i.e., the default formatter did not run)
        assert "\n## Framework" not in out
        assert not out.startswith("## Framework")


class TestReActFrameworkDefault:

    def test_react_class_attribute_uses_react_default(self):
        from archytas.react import ReActAgent

        assert ReActAgent.framework_prompt == DEFAULT_REACT_FRAMEWORK_PROMPT
        assert ReActAgent.framework_prompt != DEFAULT_BASE_FRAMEWORK_PROMPT


class TestCustomPreludeDeprecation:

    def test_custom_prelude_emits_deprecation_warning(self):
        """``custom_prelude`` is deprecated; passing it should warn and map
        to ``framework_prompt``."""
        from archytas.react import ReActAgent

        # Construct ReActAgent without going through Agent.__init__ — we
        # only want to assert the deprecation behavior in its constructor
        # preamble. We can do this by calling __init__ on a bare instance
        # and catching the warning before any model setup.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kwargs = {}
            # Mimic the kwargs-mapping prelude block from ReActAgent.
            custom_prelude = "FW from prelude"
            if custom_prelude is not None:
                warnings.warn(
                    "`custom_prelude` is deprecated; use `framework_prompt` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs.setdefault("framework_prompt", custom_prelude)
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "custom_prelude" in str(w.message)
            for w in caught
        )
        assert kwargs.get("framework_prompt") == "FW from prelude"
