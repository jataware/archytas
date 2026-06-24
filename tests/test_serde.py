"""
Tests for chat-history serialization/deserialization (serde).

These are pure unit tests: they exercise the serde machinery on hand-built
records and histories and require no model or API keys.
"""
import json
import warnings

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from archytas.chat_history import (
    ChatHistory,
    MessageRecord,
    SummaryRecord,
    SerdeSchema,
    MESSAGE_FORMAT,
    MESSAGE_RECORD_SCHEMA,
    SUMMARY_RECORD_SCHEMA,
    CHAT_HISTORY_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def build_history() -> ChatHistory:
    """A history with a mix of message types, a summary, and populated metadata."""
    history = ChatHistory()
    history.current_loop_id = 7

    r1 = history.add_message(HumanMessage(content="hello"))
    r1.token_count = 5
    r1.metadata = {"source": "user", "nested": {"k": [1, 2, 3]}}

    r2 = history.add_message(
        AIMessage(
            content="calling a tool",
            tool_calls=[{"name": "foo", "args": {"x": 1}, "id": "tc1"}],
        )
    )
    history.add_message(ToolMessage(content="tool result", tool_call_id="tc1"))

    summary = SummaryRecord(
        message=SystemMessage(content="summary of the conversation so far"),
        summarized_messages={r1.uuid, r2.uuid},
    )
    history.summaries.append(summary)

    history.set_system_message("you are a helpful assistant")
    history.set_system_preamble_text("system preamble text")
    history.set_user_preamble_text("user preamble text")

    history.tool_token_estimate = 42
    return history


def assert_json_compatible(obj) -> dict:
    """Round-trip through JSON and return the reloaded object."""
    blob = json.dumps(obj)
    return json.loads(blob)


# ---------------------------------------------------------------------------
# SerdeSchema (the reified versioning mechanism)
# ---------------------------------------------------------------------------

class TestSerdeSchema:
    def test_is_inspectable_and_jsonable(self):
        d = MESSAGE_RECORD_SCHEMA.to_dict()
        assert d == {"schema": MESSAGE_RECORD_SCHEMA.name, "version": MESSAGE_RECORD_SCHEMA.version}
        assert assert_json_compatible(d) == d

    def test_roundtrip(self):
        schema = SerdeSchema(name="archytas.Thing", version=3)
        assert SerdeSchema.from_dict(schema.to_dict()) == schema

    def test_frozen_and_hashable(self):
        # frozen dataclass -> usable in sets / as dict keys, and immutable
        s = SerdeSchema(name="a", version=1)
        assert {s, SerdeSchema(name="a", version=1)} == {s}
        with pytest.raises(Exception):
            s.version = 2  # type: ignore[misc]

    def test_compatibility(self):
        assert MESSAGE_RECORD_SCHEMA.is_compatible(MESSAGE_RECORD_SCHEMA)
        assert not MESSAGE_RECORD_SCHEMA.is_compatible(
            SerdeSchema(name=MESSAGE_RECORD_SCHEMA.name, version=MESSAGE_RECORD_SCHEMA.version + 1)
        )
        assert not MESSAGE_RECORD_SCHEMA.is_compatible(SUMMARY_RECORD_SCHEMA)

    def test_distinct_schemas(self):
        names = {
            MESSAGE_FORMAT.name,
            MESSAGE_RECORD_SCHEMA.name,
            SUMMARY_RECORD_SCHEMA.name,
            CHAT_HISTORY_SCHEMA.name,
        }
        assert len(names) == 4


# ---------------------------------------------------------------------------
# MessageRecord serde
# ---------------------------------------------------------------------------

class TestMessageRecordSerde:
    def test_carries_envelope_and_message_format_markers(self):
        record = MessageRecord(message=HumanMessage(content="hi"))
        d = record.to_dict()
        assert d["format"] == MESSAGE_RECORD_SCHEMA.to_dict()
        # inner message_format marker is independent of the envelope marker
        assert d["message_format"] == MESSAGE_FORMAT.to_dict()
        assert d["format"] != d["message_format"]

    def test_is_json_compatible(self):
        record = MessageRecord(
            message=AIMessage(content="x", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            token_count=9,
            metadata={"a": 1},
            react_loop_id=3,
        )
        reloaded = assert_json_compatible(record.to_dict())
        back = MessageRecord.from_dict(reloaded)
        assert back.uuid == record.uuid
        assert back.token_count == 9
        assert back.metadata == {"a": 1}
        assert back.react_loop_id == 3

    def test_uuid_preserved(self):
        record = MessageRecord(message=HumanMessage(content="hi"))
        back = MessageRecord.from_dict(record.to_dict())
        assert back.uuid == record.uuid

    @pytest.mark.parametrize(
        "message",
        [
            HumanMessage(content="plain human"),
            SystemMessage(content="a system prompt"),
            AIMessage(content="ai reply"),
            AIMessage(content="", tool_calls=[{"name": "foo", "args": {"x": 1}, "id": "tc1"}]),
            ToolMessage(content="the result", tool_call_id="tc1"),
        ],
    )
    def test_message_body_roundtrips(self, message):
        record = MessageRecord(message=message)
        back = MessageRecord.from_dict(record.to_dict())
        assert type(back.message) is type(message)
        assert back.message.content == message.content

    def test_tool_call_details_survive(self):
        record = MessageRecord(
            message=AIMessage(
                content="",
                tool_calls=[{"name": "foo", "args": {"x": 1, "y": "z"}, "id": "tc1"}],
            )
        )
        back = MessageRecord.from_dict(record.to_dict())
        tc = back.message.tool_calls[0]
        assert tc["name"] == "foo"
        assert tc["args"] == {"x": 1, "y": "z"}
        assert tc["id"] == "tc1"

    def test_metadata_is_deep_copied(self):
        # serialization must not alias mutable metadata back to the live record
        meta = {"nested": {"k": 1}}
        record = MessageRecord(message=HumanMessage(content="hi"), metadata=meta)
        d = record.to_dict()
        d["metadata"]["nested"]["k"] = 999
        assert record.metadata["nested"]["k"] == 1

    def test_none_scalars_roundtrip(self):
        record = MessageRecord(message=HumanMessage(content="hi"))
        assert record.token_count is None
        assert record.react_loop_id is None
        back = MessageRecord.from_dict(record.to_dict())
        assert back.token_count is None
        assert back.react_loop_id is None


# ---------------------------------------------------------------------------
# SummaryRecord serde
# ---------------------------------------------------------------------------

class TestSummaryRecordSerde:
    def test_uses_its_own_envelope_schema(self):
        summary = SummaryRecord(message=SystemMessage(content="s"))
        assert summary.to_dict()["format"] == SUMMARY_RECORD_SCHEMA.to_dict()

    def test_summarized_messages_crosses_json_as_list(self):
        summary = SummaryRecord(
            message=SystemMessage(content="s"),
            summarized_messages={"a", "b", "c"},
        )
        d = summary.to_dict()
        assert isinstance(d["summarized_messages"], list)
        # sorted for deterministic output
        assert d["summarized_messages"] == ["a", "b", "c"]

    def test_summarized_messages_restored_as_set(self):
        summary = SummaryRecord(
            message=SystemMessage(content="s"),
            summarized_messages={"a", "b"},
        )
        back = SummaryRecord.from_dict(assert_json_compatible(summary.to_dict()))
        assert isinstance(back.summarized_messages, set)
        assert back.summarized_messages == {"a", "b"}
        assert back.uuid == summary.uuid

    def test_empty_summarized_messages(self):
        summary = SummaryRecord(message=SystemMessage(content="s"))
        back = SummaryRecord.from_dict(summary.to_dict())
        assert back.summarized_messages == set()

    def test_base_class_dispatches_to_subclass(self):
        # MessageRecord.from_dict on a summary payload should produce a SummaryRecord
        summary = SummaryRecord(
            message=SystemMessage(content="s"),
            summarized_messages={"a", "b"},
        )
        back = MessageRecord.from_dict(summary.to_dict())
        assert isinstance(back, SummaryRecord)
        assert back.summarized_messages == {"a", "b"}


# ---------------------------------------------------------------------------
# ChatHistory aggregate serde
# ---------------------------------------------------------------------------

class TestChatHistorySerde:
    def test_document_shape(self):
        doc = build_history().to_dict()
        assert doc["format"] == CHAT_HISTORY_SCHEMA.to_dict()
        assert set(doc) == {
            "format",
            "metadata",
            "system_message",
            "system_preamble",
            "user_preamble",
            "raw_records",
            "summaries",
        }
        assert len(doc["raw_records"]) == 3
        assert len(doc["summaries"]) == 1

    def test_document_is_json_compatible(self):
        doc = build_history().to_dict()
        # must not raise
        assert_json_compatible(doc)

    def test_full_roundtrip_preserves_uuids(self):
        history = build_history()
        doc = assert_json_compatible(history.to_dict())
        restored = ChatHistory.from_dict(doc)

        assert [r.uuid for r in restored.raw_records] == [r.uuid for r in history.raw_records]
        assert [s.uuid for s in restored.summaries] == [s.uuid for s in history.summaries]

    def test_roundtrip_preserves_summary_sets(self):
        history = build_history()
        original = history.summaries[0].summarized_messages
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))
        assert isinstance(restored.summaries[0].summarized_messages, set)
        assert restored.summaries[0].summarized_messages == original

    def test_roundtrip_preserves_record_payloads(self):
        history = build_history()
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))

        assert restored.raw_records[0].message.content == "hello"
        assert restored.raw_records[0].token_count == 5
        assert restored.raw_records[0].metadata == {"source": "user", "nested": {"k": [1, 2, 3]}}
        assert restored.raw_records[1].message.tool_calls[0]["id"] == "tc1"
        assert restored.raw_records[2].message.tool_call_id == "tc1"

    def test_roundtrip_preserves_metadata(self):
        history = build_history()
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))
        assert restored.current_loop_id == 7
        assert restored.tool_token_estimate == 42

    def test_record_types_preserved(self):
        history = build_history()
        restored = ChatHistory.from_dict(history.to_dict())
        assert all(isinstance(r, MessageRecord) for r in restored.raw_records)
        assert all(isinstance(s, SummaryRecord) for s in restored.summaries)

    def test_empty_history_roundtrips(self):
        restored = ChatHistory.from_dict(assert_json_compatible(ChatHistory().to_dict()))
        assert restored.raw_records == []
        assert restored.summaries == []

    def test_model_is_not_required_and_not_serialized(self):
        doc = build_history().to_dict()
        # No model was attached -> model metadata is None and absent from reconstruction.
        assert doc["metadata"]["model"] is None
        restored = ChatHistory.from_dict(doc)
        assert restored.model is None


class TestSystemMessageAndPreambles:
    def test_roundtrip_preserves_system_message_and_preambles(self):
        history = build_history()
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))

        assert restored.system_message is not None
        assert restored.system_message.message.content == "you are a helpful assistant"
        assert restored.system_message.uuid == history.system_message.uuid

        assert restored.system_preamble is not None
        assert isinstance(restored.system_preamble.message, SystemMessage)
        assert restored.system_preamble.message.content == "system preamble text"
        assert restored.system_preamble.uuid == history.system_preamble.uuid
        assert restored.system_preamble.metadata == {"preamble": True}

        assert restored.user_preamble is not None
        assert isinstance(restored.user_preamble.message, HumanMessage)
        assert restored.user_preamble.message.content == "user preamble text"
        assert restored.user_preamble.uuid == history.user_preamble.uuid

    def test_unset_fields_serialize_to_null(self):
        doc = ChatHistory().to_dict()
        assert doc["system_message"] is None
        assert doc["system_preamble"] is None
        assert doc["user_preamble"] is None

    def test_null_fields_roundtrip_to_none(self):
        restored = ChatHistory.from_dict(assert_json_compatible(ChatHistory().to_dict()))
        assert restored.system_message is None
        assert restored.system_preamble is None
        assert restored.user_preamble is None

    def test_missing_keys_tolerated(self):
        # A document predating these fields (keys entirely absent) must still load.
        doc = build_history().to_dict()
        del doc["system_message"]
        del doc["system_preamble"]
        del doc["user_preamble"]
        restored = ChatHistory.from_dict(doc)
        assert restored.system_message is None
        assert restored.system_preamble is None
        assert restored.user_preamble is None
        # the rest of the document is unaffected
        assert len(restored.raw_records) == 3

    def test_setters_target_distinct_fields(self):
        # Regression: set_system_preamble_text once routed to user_preamble.
        # The two setters must populate distinct fields and not clobber each other.
        history = ChatHistory()
        history.set_system_preamble_text("sys")
        history.set_user_preamble_text("usr")

        assert history.system_preamble is not None
        assert isinstance(history.system_preamble.message, SystemMessage)
        assert history.system_preamble.message.content == "sys"

        assert history.user_preamble is not None
        assert isinstance(history.user_preamble.message, HumanMessage)
        assert history.user_preamble.message.content == "usr"

        # And the distinction survives a serde round-trip.
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))
        assert restored.system_preamble.message.content == "sys"
        assert isinstance(restored.system_preamble.message, SystemMessage)
        assert restored.user_preamble.message.content == "usr"
        assert isinstance(restored.user_preamble.message, HumanMessage)

    def test_partial_set_roundtrips(self):
        history = ChatHistory()
        history.set_system_message("only a system message")
        restored = ChatHistory.from_dict(assert_json_compatible(history.to_dict()))
        assert restored.system_message is not None
        assert restored.system_message.message.content == "only a system message"
        assert restored.system_preamble is None
        assert restored.user_preamble is None


# ---------------------------------------------------------------------------
# Schema drift / robustness
# ---------------------------------------------------------------------------

class TestSchemaDrift:
    def test_incompatible_envelope_version_warns_but_loads(self):
        doc = build_history().to_dict()
        doc["format"] = {"schema": CHAT_HISTORY_SCHEMA.name, "version": 999}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = ChatHistory.from_dict(doc)
        assert any("not compatible" in str(w.message) for w in caught)
        # still loads best-effort
        assert len(restored.raw_records) == 3

    def test_missing_schema_marker_warns_but_loads(self):
        record = MessageRecord(message=HumanMessage(content="hi"))
        d = record.to_dict()
        del d["format"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = MessageRecord.from_dict(d)
        assert any("missing" in str(w.message) for w in caught)
        assert back.uuid == record.uuid

    def test_message_format_drift_is_independent_of_envelope(self):
        # Bumping only the inner message_format must not require touching the
        # envelope schema: a record with a mismatched message_format still warns
        # on the inner marker only.
        record = MessageRecord(message=HumanMessage(content="hi"))
        d = record.to_dict()
        d["message_format"] = {"schema": MESSAGE_FORMAT.name, "version": 999}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = MessageRecord.from_dict(d)
        messages = [str(w.message) for w in caught]
        # the envelope was untouched, so the only complaint is about the body format
        assert any(MESSAGE_FORMAT.name in m and "not compatible" in m for m in messages)
        assert back.message.content == "hi"
