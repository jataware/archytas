import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import BaseArchytasModel
from ..message_schemas import ToolUseRequest
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

if TYPE_CHECKING:
    from ..agent import SystemMessage, AIMessage, ToolMessage, FunctionMessage


# Reasoning effort levels supported by each model family, keyed by model prefix pattern.
# Order within each list reflects what the model accepts.
REASONING_EFFORT_LEVELS: dict[str, list[str]] = {
    "gemini-2.5":       ["budget"],
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro": ["low", "medium", "high"],
    "gemini-3.1-flash-lite": ["minimal", "low", "medium", "high"],
}
# Preferred effort levels in priority order — the first level that the model
# supports will be selected.
PREFERRED_EFFORT_ORDER: list[str] = ["medium", "high", "low", "minimal", "budget"]

REASONING_EFFORT_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_EFFORT", None)
REASONING_BUDGET_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_BUDGET", None)
DEFAULT_REASONING_BUDGET: int = -1

class GeminiModel(BaseArchytasModel):
    DEFAULT_MODEL = "gemini-2.5-flash"
    api_key: str


    MODEL_PROMPT_INSTRUCTIONS = """
Before you call EACH and ANY tool, you MUST add a text block to the output a plain-text 1-3 sentence explanation of reasoning why a particular tool or code generation was selected for use. This will be shown to the user so the user can keep track of what you are up to, so be succinct, polite, and helpful.
This must be done for EVERY tool call. When preparing to use a tool, you must explicitly address the user. Your internal thinking is for you; your text output is for the user. Always write a short text message directly to the user explaining what you are about to do before the actual tool invocation.

IMPORTANT: You possess an internal thinking/reasoning process, but the user CANNOT see this as direct communication. Therefore, before generating ANY tool call, you MUST output a standard, top-level text response (e.g., a "type": "text" block) containing a 1-3 sentence conversational update for the user. Do not bury your explanation to the user inside your internal thinking block.

Remember your internal reasoning and thoughts are communicated to the user. You must communicate with the user by including a text block in your chat generation output.
"""

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = self.config.api_key
        if auth_token:
            self.api_key = auth_token
        else:
            raise AuthenticationError("Gemini API key not provided.")

    def _get_supported_reasoning_efforts(self, model_name: str) -> list[str] | None:
        """Return the list of supported reasoning effort levels for the given model, or None if unsupported."""
        model_lower = model_name.lower()
        # Check prefixes longest-first so e.g. "gemini-3.1-flash-lite" matches before "gemini-3.1".
        for prefix in sorted(REASONING_EFFORT_LEVELS, key=len, reverse=True):
            if model_lower.startswith(prefix):
                return REASONING_EFFORT_LEVELS[prefix]
        return None

    def _get_thinking_config(self, model_name: str) -> dict[str, object] | None:
        """
        Build the thinking kwargs dict for ChatGoogleGenerativeAI, or None if the model
        doesn't support thinking.

        For "budget" effort, returns include_thoughts + thinking_budget_token_limit.
        For named levels (minimal/low/medium/high), returns include_thoughts + thinking_level.
        """
        supported = self._get_supported_reasoning_efforts(model_name)
        if supported is None:
            return None

        # Determine the effort level: env var override, then preferred order fallback.
        effort: str | None = None
        if REASONING_EFFORT_REQUEST and REASONING_EFFORT_REQUEST in supported:
            effort = REASONING_EFFORT_REQUEST
        else:
            for candidate in PREFERRED_EFFORT_ORDER:
                if candidate in supported:
                    effort = candidate
                    break

        if effort is None:
            # Model is in the table but no usable effort level found; just enable thoughts.
            return {"include_thoughts": True}

        config: dict[str, object] = {"include_thoughts": True}
        if effort == "budget":
            budget = DEFAULT_REASONING_BUDGET
            if REASONING_BUDGET_REQUEST is not None:
                try:
                    budget = int(REASONING_BUDGET_REQUEST)
                except ValueError:
                    pass
            config["thinking_budget_token_limit"] = budget
        else:
            config["thinking_level"] = effort

        return config

    def initialize_model(self, **kwargs):
        model_name = self.config.model_name or self.DEFAULT_MODEL
        model_kwargs = dict(
            model=model_name,
            api_key=self.api_key,
        )
        thinking = self._get_thinking_config(model_name)
        if thinking is not None:
            model_kwargs.update(thinking)
        return ChatGoogleGenerativeAI(**model_kwargs)

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        # Gemini doesn't accept a temperature keyword on invoke
        kwargs.pop("temperature")
        return  await super().ainvoke(input, config=config, stop=stop, **kwargs)

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        if "flash" in model_name:
            return 1_048_576
        elif "pro" in model_name:
            # gemini 2.0/2.5 have 2M context, others do not
            if "2." in model_name:
                return 2_097_152
            else:
                return 1_048_576

    def _preprocess_messages(self, messages):
        from langchain_core.messages import AIMessage
        from ..agent import SystemMessage
        output = []
        system_messages = []
        # (SystemMessage subsumes ContextMessage and the legacy AutoContextMessage
        # via inheritance, so a single isinstance check covers all three.)
        for message in messages:
            if isinstance(message, SystemMessage):
                system_messages.append(message.content)
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any(('400 API key not valid' in arg for arg in error.args)):
                raise AuthenticationError("API key invalid.") from error
            elif any(('exceeds the maximum number of tokens allowed' in arg for arg in error.args)):
                raise ContextWindowExceededError("Context window maximum tokens exceeded.") from error
        raise ExecutionError(*error.args) from error

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        from ..agent import AgentResponse
        content = response_message.content
        tool_calls = response_message.tool_calls

        text = ""
        metadata = {}
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract all content blocks with source labels
            labeled_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = item.get("type")
                if block_type == "text" and len(item.get("text") or "") > 0:
                    labeled_parts.append(str(item['text']))
                elif block_type == "reasoning":
                    metadata["reasoning"] = item.get("reasoning")
                elif block_type == "thinking":
                    metadata["thinking"] = item.get("thinking")
            if labeled_parts:
                text = "\n".join(labeled_parts)

        return AgentResponse(text=text, tool_calls=tool_calls)
