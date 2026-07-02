"""
Tests for issue #85: summarization stalls when a single message is larger than
the summarization threshold.

All tests run offline against FakeModel (tokens ~= chars // 4).
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from archytas.chat_history import ChatHistory, MessageRecord
from archytas.summarizers import (
    TRUNCATION_PLACEHOLDER,
    default_history_summarizer,
    fit_records_to_context,
    get_records_up_to_threshold,
    middle_truncate_text,
    shrink_record_for_summarization,
)

from .fake_model import FakeModel


def make_history(model, messages_with_tokens) -> ChatHistory:
    """Build a ChatHistory whose records carry preset token counts."""
    history = ChatHistory(model=model)
    for message, tokens in messages_with_tokens:
        record = history.add_message(message)
        record.token_count = tokens
    return history


def text_of_tokens(tokens: int) -> str:
    """Text sized so FakeModel estimates it at roughly `tokens` tokens."""
    return "x" * (tokens * 4)


class TestRecordSelectionFallback:

    @pytest.mark.asyncio
    async def test_single_oversized_record_is_selected(self):
        """A lone record larger than the threshold is selected instead of
        returning an empty set (the issue #85 stall)."""
        model = FakeModel()
        history = make_history(model, [
            (HumanMessage(content=text_of_tokens(5000)), 5000),
            (AIMessage(content="ok"), 20),
        ])

        selected = await get_records_up_to_threshold(history, model, token_threshold=1000)

        assert len(selected) == 1
        assert selected[0].uuid == history.raw_records[0].uuid

    @pytest.mark.asyncio
    async def test_oversized_tool_message_keeps_pair_together(self):
        """When the oversized record is a ToolMessage, its calling AIMessage
        and sibling ToolMessages are included so the pair is never split."""
        model = FakeModel()
        tool_calls = [
            {"id": "call_1", "name": "big_tool", "args": {}, "type": "tool_call"},
            {"id": "call_2", "name": "big_tool", "args": {}, "type": "tool_call"},
        ]
        history = make_history(model, [
            (AIMessage(content="calling tool", tool_calls=tool_calls), 50),
            (ToolMessage(content=text_of_tokens(5000), tool_call_id="call_1"), 5000),
            (ToolMessage(content="small result", tool_call_id="call_2"), 10),
            (HumanMessage(content="next question"), 20),
        ])

        selected = await get_records_up_to_threshold(history, model, token_threshold=1000)

        assert [record.uuid for record in selected] == [
            record.uuid for record in history.raw_records[:3]
        ]

    @pytest.mark.asyncio
    async def test_history_under_threshold_still_selects_nothing(self):
        """The fallback must not fire when the history fits under the
        threshold; that path intentionally summarizes nothing."""
        model = FakeModel()
        history = make_history(model, [
            (HumanMessage(content="hello"), 100),
            (AIMessage(content="hi"), 100),
        ])

        selected = await get_records_up_to_threshold(history, model, token_threshold=1000)

        assert selected == []

    @pytest.mark.asyncio
    async def test_normal_threshold_selection_unchanged(self):
        """The regular path (several records under the threshold) behaves as
        before: select the prefix that fits."""
        model = FakeModel()
        history = make_history(model, [
            (HumanMessage(content=f"message {i}"), 300) for i in range(5)
        ])

        selected = await get_records_up_to_threshold(history, model, token_threshold=1000)

        assert len(selected) == 3


class TestMiddleTruncation:

    def test_keeps_head_and_tail(self):
        text = "HEAD" + ("m" * 10000) + "TAIL"
        truncated = middle_truncate_text(text, keep_chars=1000)
        assert truncated.startswith("HEAD")
        assert truncated.endswith("TAIL")
        assert "removed from the middle" in truncated
        assert len(truncated) < len(text)

    def test_short_text_untouched(self):
        assert middle_truncate_text("short", keep_chars=1000) == "short"

    def test_zero_budget_gives_placeholder(self):
        assert middle_truncate_text("anything", keep_chars=0) == TRUNCATION_PLACEHOLDER

    @pytest.mark.asyncio
    async def test_shrink_record_fits_target_and_preserves_original(self):
        model = FakeModel()
        original_text = "HEAD" + ("m" * 40000) + "TAIL"
        record = MessageRecord(message=HumanMessage(content=original_text), token_count=10000)

        shrunk = await shrink_record_for_summarization(record, model, target_tokens=1000)

        assert shrunk.uuid == record.uuid
        assert shrunk.token_count <= 1000
        assert "removed from the middle" in shrunk.message.content
        # Original record is left completely untouched.
        assert record.message.content == original_text
        assert record.token_count == 10000

    @pytest.mark.asyncio
    async def test_non_string_content_replaced_with_placeholder(self):
        model = FakeModel()
        content = [{"type": "text", "text": "m" * 40000}]
        record = MessageRecord(message=HumanMessage(content=content), token_count=10000)

        shrunk = await shrink_record_for_summarization(record, model, target_tokens=1000)

        assert shrunk.message.content == TRUNCATION_PLACEHOLDER
        assert record.message.content == content

    @pytest.mark.asyncio
    async def test_tiny_target_replaced_with_placeholder(self):
        model = FakeModel()
        record = MessageRecord(message=HumanMessage(content="m" * 40000), token_count=10000)

        shrunk = await shrink_record_for_summarization(record, model, target_tokens=10)

        assert shrunk.message.content == TRUNCATION_PLACEHOLDER


class TestFitRecordsToContext:

    @pytest.mark.asyncio
    async def test_records_within_budget_pass_through(self):
        model = FakeModel(context_window=100000)
        records = [
            MessageRecord(message=HumanMessage(content="small"), token_count=10),
        ]
        fitted = await fit_records_to_context(records, model)
        assert fitted[0] is records[0]

    @pytest.mark.asyncio
    async def test_oversized_record_truncated_to_fit(self):
        # context 2000 -> budget = max(2000 - 8192, 1000) = 1000
        model = FakeModel(context_window=2000)
        small = MessageRecord(message=HumanMessage(content="small question"), token_count=10)
        huge = MessageRecord(
            message=HumanMessage(content=text_of_tokens(5000)), token_count=5000,
        )

        fitted = await fit_records_to_context([small, huge], model)

        assert fitted[0] is small
        assert fitted[1] is not huge
        assert fitted[1].uuid == huge.uuid
        total = sum(record.token_count for record in fitted)
        assert total <= 1000
        # Original untouched.
        assert huge.token_count == 5000


class TestEndToEndSummarization:

    @pytest.mark.asyncio
    async def test_oversized_message_gets_summarized(self):
        """The full path: an oversized record is selected, truncated for the
        request, summarized, and thereafter excluded from outgoing history."""
        model = FakeModel(context_window=2000)  # threshold = 1000
        history = make_history(model, [
            (HumanMessage(content="HEAD" + text_of_tokens(5000) + "TAIL"), 5001),
        ])
        original_content = history.raw_records[0].message.content

        recordset = await get_records_up_to_threshold(history, model, token_threshold=1000)
        assert recordset, "oversized record must be selected for summarization"

        await default_history_summarizer(
            chat_history=history, agent=None, recordset=recordset, model=model,
        )

        assert len(history.summaries) == 1
        summary = history.summaries[0]
        assert summary.summarized_messages == {history.raw_records[0].uuid}

        # The original record was not mutated...
        assert history.raw_records[0].message.content == original_content
        # ...but is excluded from the outgoing history now that it is summarized.
        outgoing = await history.records()
        assert history.raw_records[0].uuid not in [record.uuid for record in outgoing]

        # The request actually sent to the model was truncated to fit.
        sent = model._model.calls[0]
        user_prompt = sent[-1].content
        assert "removed from the middle" in user_prompt
        assert len(user_prompt) < len(original_content)
