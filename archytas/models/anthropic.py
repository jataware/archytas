import json
import os
import re
import logging
from functools import lru_cache

from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError, BadRequestError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_anthropic.llms import AnthropicLLM
from pydantic import BaseModel as PydanticModel, Field

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError


def _prompt_cache_disabled() -> bool:
    """Env-var escape hatch for the Phase 4 cache_control markers.

    Set ARCHYTAS_DISABLE_PROMPT_CACHE=1 to bypass cache_control injection
    (used by the measurement harness to compare before/after behavior).
    """
    val = os.environ.get("ARCHYTAS_DISABLE_PROMPT_CACHE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _attach_cache_control(message: BaseMessage) -> BaseMessage:
    """Return a copy of `message` with cache_control:ephemeral on its last text block.

    For string-content messages, converts to the content-block form expected
    by langchain-anthropic (one text block with cache_control). For
    list-content messages, augments the last text block in place on a copy.
    Non-text-bearing messages are returned unchanged.
    """
    marker = {"type": "ephemeral"}
    try:
        copied = message.model_copy(deep=True)
    except Exception:
        # Fall back: leave the message unchanged rather than crashing the request.
        return message

    content = copied.content
    if isinstance(content, str):
        if not content:
            return message
        copied.content = [
            {"type": "text", "text": content, "cache_control": marker}
        ]
        return copied

    if isinstance(content, list):
        # Find the last text block and augment it; if none, leave unchanged.
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                block["cache_control"] = marker
                return copied
        return message

    return message

class DummyTool(PydanticModel):
    """
    Dummy Tool
    """
    input: str = Field(..., description="input")


class AnthropicModel(BaseArchytasModel):
    DEFAULT_MODEL: str = "claude-3-5-sonnet-latest"
    api_key: str = ""
    tool_name_map: dict
    rev_tool_name_map: dict

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.tool_name_map = {}
        self.rev_tool_name_map = {}

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        else:
            self.api_key = self.config.api_key or ""
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        max_tokens = None
        if self.config.model_extra:
            max_tokens = self.config.model_extra.get('max_tokens', None)
        return ChatAnthropic(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.api_key,
            max_tokens=max_tokens
        )

    def _preprocess_messages(self, messages):
        from ..agent import ContextMessage
        output = []

        system_messages = []
        # Combine all system/context messages into a single initial system message.
        # (ContextMessage subsumes the legacy AutoContextMessage via inheritance,
        # so no separate case is needed for it.)
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage():
                    system_messages.append(message.content)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))

        # Prompt-caching marker (plan §4.4): attach cache_control:ephemeral to
        # the last PERSISTED message in the outgoing list. Tail injections
        # (state pairs + instruction) are beyond the cacheable prefix and
        # must not be marked. The boundary is communicated by
        # self.tail_injection_count, set by Agent.execute() just before
        # invoke. System-type messages in the persisted portion have been
        # collapsed into the leading SystemMessage; tail messages are never
        # system-type, so their count is unchanged by the collapse.
        if not _prompt_cache_disabled():
            tail_count = getattr(self, "tail_injection_count", 0) or 0
            last_persisted_idx = len(output) - tail_count - 1
            # Skip marking the lone collapsed SystemMessage (idx 0) — caching
            # the system prompt alone is usually not worth the cache breakpoint.
            if last_persisted_idx > 0:
                output[last_persisted_idx] = _attach_cache_control(output[last_persisted_idx])

        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        # TODO: Retry with delay on rate limit errors?
        # elif isinstance(error, RateLimitError):
        #     raise
        elif isinstance(error, BadRequestError) and "prompt is too long" in error.message:
            sent = None
            maximum = None
            try:
                body = error.response.json().get("error", {})
                if "message" in body:
                    match: re.Match = re.search(r'prompt is too long: (\d+) .* (\d+) maximum', body["message"])
                    if match:
                        sent, maximum = match.groups()
            finally:
                raise ContextWindowExceededError(*error.args, sent=sent, maximum=maximum) from error
        else:
            # if self.last_messages:
            #     message_output = [msg.model_dump() for msg in self.last_messages]
            #     logging.warning(
            #         "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
            #         json.dumps(message_output, indent=2)
            #     )
            raise

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        # TODO: This is accurate for all current models (as of 2025-02-27), for all Haiku and Sonnet models. Seems like
        # new models would have a new name if the context window changes. Hopefully in the future, this value can be
        # retrieved programatically.
        if model_name and ("haiku" in model_name or "sonnet" in model_name):
            return 200_000
        else:
            return None
