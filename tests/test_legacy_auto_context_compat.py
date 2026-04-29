"""
Regression tests for the deprecation-shim compatibility layer over the legacy
auto-context API (plan Phase 3).

These tests cover the Beaker access pattern specifically — where downstream
code reaches into `chat_history.auto_context_message` to set `_model`, clear
`content`, and call `update_content()` manually. That pattern was documented
in pre-relocation versions of archytas and must keep working (with a
DeprecationWarning) until the shim is eventually removed.
"""
import warnings

import pytest

from archytas.chat_history import (
    AutoContextMessage,
    ChatHistory,
    InstructionSource,
    _AutoContextMessageShim,
)


class TestAutoContextMessageImport:
    """The class itself must remain importable from all historical locations."""

    def test_importable_from_chat_history(self):
        from archytas.chat_history import AutoContextMessage as A1
        assert A1 is AutoContextMessage

    def test_importable_from_agent(self):
        # agent.py re-exports AutoContextMessage; downstream code may import from either.
        from archytas.agent import AutoContextMessage as A2
        assert A2 is AutoContextMessage


class TestAutoContextMessageDirectConstruction:
    """Constructing AutoContextMessage directly should warn but still work."""

    def test_direct_construction_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            msg = AutoContextMessage(
                default_content="x",
                content_updater=lambda: "x",
            )
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "AutoContextMessage is deprecated" in str(w.message)
                for w in caught
            ), "direct construction must emit a DeprecationWarning"
        # The object itself should still behave like before — content set.
        assert msg.content == "x"


class TestAutoContextMessageShim:
    """The shim returned by `chat_history.auto_context_message` must preserve
    the attribute surface that Beaker (and similar consumers) reach for."""

    def _make_history_with_instruction(self) -> ChatHistory:
        ch = ChatHistory()
        ch.instruction = InstructionSource(
            default_content="initial content",
            content_updater=lambda: "updated content",
            auto_update=True,
        )
        return ch

    def test_accessor_returns_shim_when_instruction_set(self):
        ch = self._make_history_with_instruction()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shim = ch.auto_context_message
        assert shim is not None
        assert isinstance(shim, _AutoContextMessageShim)

    def test_accessor_returns_none_when_no_instruction(self):
        ch = ChatHistory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert ch.auto_context_message is None

    def test_accessor_emits_deprecation_warning(self):
        ch = self._make_history_with_instruction()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = ch.auto_context_message
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "auto_context_message is deprecated" in str(w.message)
                for w in caught
            )

    def test_shim_model_getset_routes_to_instruction(self):
        """Beaker does: `chat_history.auto_context_message._model = agent.model`"""
        ch = self._make_history_with_instruction()
        sentinel = object()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shim = ch.auto_context_message
            shim._model = sentinel
        assert ch.instruction._model is sentinel

    def test_shim_content_getset_routes_to_instruction(self):
        """Beaker does: `chat_history.auto_context_message.content = ""`"""
        ch = self._make_history_with_instruction()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shim = ch.auto_context_message
            assert shim.content == "initial content"
            shim.content = ""
        assert ch.instruction.current_content == ""

    @pytest.mark.asyncio
    async def test_shim_update_content_routes_to_instruction(self):
        """Beaker does: `await chat_history.auto_context_message.update_content()`"""
        ch = self._make_history_with_instruction()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shim = ch.auto_context_message
            # Match Beaker's setup: force a hash mismatch by zeroing content first.
            shim.content = ""
            await shim.update_content()
        # The updater returns "updated content"; verify it took effect.
        assert ch.instruction.current_content == "updated content"

    @pytest.mark.asyncio
    async def test_full_beaker_access_pattern(self):
        """End-to-end reproduction of the Beaker `set_agent_history` flow."""
        ch = self._make_history_with_instruction()
        fake_model_ref = object()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Three distinct attribute accesses — matches Beaker's code exactly.
            ch.auto_context_message._model = fake_model_ref
            ch.auto_context_message.content = ""
            await ch.auto_context_message.update_content()

        assert ch.instruction._model is fake_model_ref
        assert ch.instruction.current_content == "updated content"


class TestAutoUpdateContextShim:
    """The `auto_update_context` flag must remain get/set-able."""

    def test_get_returns_instruction_auto_update(self):
        ch = ChatHistory()
        ch.instruction = InstructionSource(default_content="x", auto_update=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert ch.auto_update_context is True
            ch.instruction.auto_update = False
            assert ch.auto_update_context is False

    def test_get_returns_false_when_no_instruction(self):
        ch = ChatHistory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert ch.auto_update_context is False

    def test_set_routes_to_instruction(self):
        ch = ChatHistory()
        ch.instruction = InstructionSource(default_content="x", auto_update=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ch.auto_update_context = False
        assert ch.instruction.auto_update is False

    def test_accessor_emits_deprecation_warning(self):
        ch = ChatHistory()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = ch.auto_update_context
            assert any(
                issubclass(w.category, DeprecationWarning)
                for w in caught
            )


class TestUpdateAutoContextShim:
    """The `update_auto_context()` method must remain callable."""

    @pytest.mark.asyncio
    async def test_method_delegates_to_instruction(self):
        ch = ChatHistory()
        captured = []
        ch.instruction = InstructionSource(
            default_content="before",
            content_updater=lambda: captured.append("called") or "after",
            auto_update=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await ch.update_auto_context()
        assert captured == ["called"]
        assert ch.instruction.current_content == "after"

    @pytest.mark.asyncio
    async def test_method_emits_deprecation_warning(self):
        ch = ChatHistory()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await ch.update_auto_context()
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "update_auto_context" in str(w.message)
                for w in caught
            )
