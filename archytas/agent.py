import asyncio
import inspect
import logging
import os
import uuid
from tenacity import (
    before_sleep_log,
    retry as tenacity_retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from typing import Callable, ContextManager, Any, Optional
from uuid import UUID


from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall

from .chat_history import ChatHistory, ContextMessage, AutoContextMessage, AgentResponse, InstructionSource
from .exceptions import AuthenticationError, ExecutionError, ModelError
from .models.base import BaseArchytasModel
from .prompt import (
    DEFAULT_BASE_FRAMEWORK_PROMPT,
    DEFAULT_HEADER_FORMATTER,
    HeaderFormatter,
    PromptSection,
    assemble_prompt,
)
from .utils import ensure_async


# Sentinel used to distinguish an explicit `prompt=` pass-through (which
# maps to `custom_prompt`/full override) from the default flow (which runs
# the section assembler).
_PROMPT_UNSET = object()

logger = logging.getLogger(__name__)

# Placeholder text used for the synthesized AIMessage that carries the fabricated
# statetool tool_calls (plan §4.2.1). Per-provider override may be appropriate if
# a specific provider proves incompatible with this default.
STATE_INJECTION_PLACEHOLDER_TEXT = "Fetching system state"


def retry(fn):
    def is_openai_error(err):
        import openai
        # Don't retry auth errors. Assume to be permanent without fixing auth.
        if isinstance(err, openai.AuthenticationError):
            return False
        return isinstance(
            err,
            (
                openai.APITimeoutError,
                openai.APIError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            )
        )

    return tenacity_retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(is_openai_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(fn)


logger = logging.getLogger(__name__)


def cli_spinner():
    from rich.live import Live
    from rich.spinner import Spinner
    return Live(
        Spinner("dots", speed=2, text="thinking..."),
        refresh_per_second=30,
        transient=True,
    )


class no_spinner:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Agent:
    chat_history: ChatHistory

    # Default Framework section text. Subclasses override this class
    # attribute to replace the Framework body for all instances of that
    # subclass; instance-level overrides go through the constructor's
    # `framework_prompt=` kwarg.
    framework_prompt: str = DEFAULT_BASE_FRAMEWORK_PROMPT

    def __init__(
        self,
        *,
        model: BaseArchytasModel,
        prompt=_PROMPT_UNSET,
        custom_prompt: str | None = None,
        framework_prompt: str | None = None,
        header_formatter: HeaderFormatter | None = None,
        api_key: str | None = None,
        spinner: Callable[[], ContextManager] | None = cli_spinner,
        rich_print: bool = True,
        verbose: bool = False,
        messages: Optional[list[BaseMessage]] | None = None,
        temperature: float = 0.0,
    ):
        """
        Agent class for managing communication with Language Models mediated by Langchain

        Args:
            model (BaseArchytasModel): The model to use. Defaults to OpenAIModel(model_name="gpt-4o").
            prompt (str, optional): DEPRECATED back-compat alias. When passed
                explicitly, treated as ``custom_prompt`` (full override that
                bypasses the section assembler). Prefer ``custom_prompt`` for
                new code.
            custom_prompt (str, optional): Full system-prompt override.
                Bypasses the assembler entirely; the value is used verbatim
                as the system message. Defaults to None.
            framework_prompt (str, optional): Per-instance override of the
                Framework section body. Replaces the class-level
                ``framework_prompt`` for this instance only. Defaults to None
                (class attribute used).
            header_formatter (callable, optional): Callable taking an
                optional section name and returning either a header string or
                None to omit the header. Defaults to ``DEFAULT_HEADER_FORMATTER``.
            api_key (str, optional): The LLM provider API key to use. Defaults to None. If None, the provider will use the default environment variable (e.g. OPENAI_API_KEY).
            spinner ((fn -> ContextManager) | None, optional): A function that returns a context manager that is run every time the LLM is generating a response. Defaults to cli_spinner which is used to display a spinner in the terminal.
            rich_print (bool, optional): Whether to use rich to print messages. Defaults to True. Can also be set via the DISABLE_RICH_PRINT environment variable.
            verbose (bool, optional): Expands the debug output. Includes full query context on requests to the LLM. Defaults to False.
            messages (list[BaseMessage], optional): A list of messages to initialize the agent's conversation history with. Defaults to an empty list.

        Raises:
            Exception: If no API key is given.
        """
        self.rich_print = bool(
            rich_print and not os.environ.get("DISABLE_RICH_PRINT", False)
        )
        self.verbose = verbose
        if not isinstance(model, BaseArchytasModel):
            # Importing OpenAI is slow, so limit import to only when it is needed.
            from .models.openai import OpenAIModel
            model_name = model if isinstance(model, str) else None
            self.model = OpenAIModel({"api_key": api_key, "model_name": model_name})
        else:
            self.model = model

        # Back-compat: an explicit `prompt=` is treated as `custom_prompt`.
        # The default-default (no caller passes anything) falls through to
        # the assembler.
        if prompt is not _PROMPT_UNSET and custom_prompt is None:
            custom_prompt = prompt
        self.custom_prompt = custom_prompt
        if framework_prompt is not None:
            self.framework_prompt = framework_prompt
        self.header_formatter: HeaderFormatter = (
            header_formatter if header_formatter is not None else DEFAULT_HEADER_FORMATTER
        )

        self.chat_history = ChatHistory(messages)
        self.chat_history.set_system_message(SystemMessage(content=self.build_system_prompt()))
        if spinner is not None and self.rich_print:
            self.spinner = spinner
        else:
            self.spinner = no_spinner

        self.temperature = temperature
        self.post_execute_task = None

        # Statetool registry (plan §4.2). Populated by subclasses (e.g.
        # ReActAgent extracts statetools from its tools list after
        # make_tool_dict runs). Map of tool_name -> tool callable.
        self.statetools: dict[str, Callable] = {}

    def get_prompt_sections(self) -> list[PromptSection]:
        """Return the ordered list of ``PromptSection``s composing the system
        prompt.

        Default returns the Framework section (sourced from
        ``self.framework_prompt``) followed by the Model section (sourced
        from ``self.model.MODEL_PROMPT_INSTRUCTIONS``). Subclasses should
        call ``super().get_prompt_sections()`` and append their own
        sections, or filter/reorder by ``role`` for structural
        manipulation.
        """
        sections: list[PromptSection] = []

        framework_text = (self.framework_prompt or "").strip()
        if framework_text:
            sections.append(
                PromptSection(body=framework_text, name="Framework", role="framework")
            )

        model_text = getattr(self.model, "MODEL_PROMPT_INSTRUCTIONS", "") or ""
        model_text = model_text.strip()
        if model_text:
            sections.append(
                PromptSection(body=model_text, name="Model", role="model")
            )

        return sections

    def build_system_prompt(self) -> str:
        """Build the final system prompt string.

        Returns ``self.custom_prompt`` verbatim if set; otherwise assembles
        the sections returned by ``get_prompt_sections()`` using
        ``self.header_formatter``.
        """
        if self.custom_prompt is not None:
            return self.custom_prompt
        return assemble_prompt(self.get_prompt_sections(), self.header_formatter)

    def update_system_prompt(self) -> None:
        """Rebuild the system prompt and write it to chat history.

        Call this after changing anything that affects the assembled
        prompt (e.g., ``framework_prompt``, model, sections returned by an
        override of ``get_prompt_sections``).
        """
        self.chat_history.set_system_message(
            SystemMessage(content=self.build_system_prompt())
        )

    def print(self, *args, **kwargs):
        if self.rich_print:
            # Defer rich print import until it is needed
            from rich import print as rprint
            rprint(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def debug(self, event_type: str, content: Any) -> None:
        """
        Debug handler

        Function at this level so it can be overridden in a subclass.
        """
        logger.debug(event_type, content)

    def log(self, event_type: str, content: Any) -> None:
        logger.info(event_type, content)

    def set_openai_key(self, key):
        import openai
        from .models.openai import OpenAIModel

        openai.api_key = key
        if isinstance(self.model, OpenAIModel):
            self.model.auth(api_key=key)


    async def all_messages(self) -> list[BaseMessage]:
        return await self.chat_history.all_messages()

    def add_context(self, context: str, *, lifetime: int | None = None) -> int:
        """
        Inject a context message to the agent's conversation.

        Useful for providing the agent with information relevant to the current conversation, e.g. tool state, environment info, etc.
        If a lifetime is specified, the context message will automatically be deleted from the chat history after that many steps.
        A context message can be deleted manually by calling clear_context() with the id of the context message.

        Args:
            context (str): The context to add to the agent's conversation.
            lifetime (Optional[int]): DEPRECATED. Ignored.

        Returns:
            int: The id of the context message.
        """
        if lifetime is not None:
            raise DeprecationWarning("Context message lifetimes have deprecated and will be ignored.")
        context_message = ContextMessage(
            content=context,
            lifetime=None,
        )
        record = self.chat_history.add_message(context_message)
        uuid = UUID(hex=record.uuid, version=4)
        return uuid.int

    def clear_context(self, id: int) -> None:
        """
        Remove a single context message from the agent's conversation.

        Args:
            id (int): The id of the context message to remove.
        """
        from .chat_history import MessageRecord
        idx: int = -1
        for index, record in enumerate(self.chat_history.raw_records):
            if isinstance(record, MessageRecord) and isinstance(record.message, ContextMessage):
                if UUID(hex=record.uuid, version=4).int == id:
                    idx = index
                    break
        if idx > -1:
            del self.chat_history.raw_records[idx]

    def clear_all_context(self) -> None:
        """Remove all context messages from the agent's conversation."""
        from .chat_history import MessageRecord
        to_remove = []
        for index, record in enumerate(self.chat_history.raw_records):
            if isinstance(record, MessageRecord) and isinstance(record.message, ContextMessage):
                to_remove.append(index)
        # Remove from the back to front so that removals don't change the indexs of subsequent items
        for idx in reversed(sorted(to_remove)):
            del self.chat_history.raw_records[idx]

    def set_auto_context(
        self,
        default_content: str,
        content_updater: Callable[[], str] | None = None,
        auto_update: bool = True,
    ):
        """
        Register dynamic behavioral guidance that is automatically injected as an
        XML-tagged HumanMessage at the tail of the outgoing message list on every
        `Agent.execute()` call. The content is never persisted to chat history,
        so per-turn updates do not disturb the cacheable prefix of the
        conversation.

        Args:
            default_content (str): The content used when the updater has not yet
                run, cannot be run, or returns a falsy value.
            content_updater (callable): A function/lambda (sync or async) taking
                no arguments and returning a string. The returned string becomes
                the new instruction content on the next call.
            auto_update (bool): If True (default), the updater is invoked on
                every `execute()` call. If False, the instruction content remains
                at `default_content` unless
                `chat_history.instruction.update_content()` is called manually.
        """
        self.chat_history.instruction = InstructionSource(
            default_content=default_content,
            content_updater=content_updater,
            auto_update=auto_update,
            model=self.model,
        )

    async def handle_message(self, message: BaseMessage):
        """Appends a message to the message list and executes."""
        self.chat_history.add_message(message)
        return await self.execute()

    async def query(self, message: str) -> str:
        """Send a user query to the agent. Returns the agent's response"""
        return await self.handle_message(HumanMessage(content=message))

    async def observe(self, observation: str, tool_name: str) -> str:
        """Send a system/tool observation to the agent. Returns the agent's response"""
        return await self.handle_message(FunctionMessage(type="function", content=observation, name=tool_name))

    async def inspect(self, query: str) -> str:
        """Send one-off system query that is not recorded in history"""
        return await self.execute([HumanMessage(content=query)])

    async def error(self, error: BaseMessage | str, drop_error: bool = True) -> str:
        """
        Send an error message to the agent. Returns the agent's response.

        Args:
            error (str): The error message to send to the agent.
            drop_error (bool, optional): If True, the error message and LLMs bad input will be dropped from the chat history. Defaults to `True`.
        """
        if not isinstance(error, BaseMessage):
            error = AIMessage(content=error)
        result = await self.handle_message(error)
        return result

    async def post_execute(self):
        if await self.chat_history.needs_summarization(model=self.model):
            await self.chat_history.summarize_history(agent=self)

    async def build_tail_injections(self) -> list[BaseMessage]:
        """
        Assemble the transient, per-call messages appended to the outgoing list
        immediately before the model is invoked.

        These messages are never persisted into chat history; they exist only
        for the current `execute()` call. Ordering: state-tool pairs first
        (§4.2), then the instruction block (§4.3). The model sees environment
        state before any behavioral guidance.

        Override `build_state_injection` to customize the layout of state
        pairs. Override this method to customize the overall tail assembly.
        """
        tail: list[BaseMessage] = []

        # --- State tool injection (Phase 2 / plan §4.2) ---
        # TODO: Deduplication between LLM-initiated statetool calls (visible
        # in recent history) and framework-injected fabricated pairs was
        # considered and intentionally deferred for v1. If duplicates become
        # noisy, revisit here: either skip injection when a ToolMessage for
        # this statetool appears within the last N records, or similar.
        fired: list[tuple[str, str]] = []
        for tool_name, tool_fn in self.statetools.items():
            if getattr(tool_fn, "_disabled", False):
                continue
            condition = getattr(tool_fn, "_statetool_condition", None)
            if condition is None:
                continue
            try:
                should_fire = await self._evaluate_statetool_condition(
                    condition, tool_name, tool_fn,
                )
            except Exception as err:
                logger.warning(
                    "Statetool %r condition raised %r; skipping injection.",
                    tool_name, err,
                )
                continue
            if not should_fire:
                continue
            try:
                output = await self._run_statetool(tool_fn, tool_name)
            except Exception as err:
                logger.warning(
                    "Statetool %r body raised %r; skipping injection.",
                    tool_name, err,
                )
                continue
            fired.append((tool_name, output))

        if fired:
            state_messages = await self.build_state_injection(fired)
            tail.extend(state_messages)
            if self.verbose:
                self.debug(
                    event_type="state_injection",
                    content={"fired": [name for name, _ in fired]},
                )

        # --- Instruction block (Phase 1 / plan §4.3) ---
        instruction = self.chat_history.instruction
        if instruction is not None:
            if instruction.auto_update:
                await instruction.update_content()
            instruction_msg = instruction.to_message()
            if instruction_msg is not None:
                tail.append(instruction_msg)

        return tail

    async def build_state_injection(
        self,
        fired: list[tuple[str, str]],
    ) -> list[BaseMessage]:
        """
        Default state-injection builder (plan §4.2.3).

        Given an ordered list of (tool_name, tool_output) pairs for statetools
        whose conditions fired this turn, return the messages to insert at
        the tail. The default bundles everything into one fabricated
        `AIMessage` with N `tool_calls` (placeholder text
        `STATE_INJECTION_PLACEHOLDER_TEXT`) followed by N `ToolMessage`s.

        Override to customize the layout — e.g. split into one AI/Tool pair
        per statetool, or reshape for a model that rejects bundled calls.
        """
        if not fired:
            return []

        tool_calls: list[dict] = []
        tool_messages: list[BaseMessage] = []
        for tool_name, output in fired:
            call_id = f"call_{uuid.uuid4().hex[:16]}"
            tool_calls.append({
                "id": call_id,
                "name": tool_name,
                "args": {},
                "type": "tool_call",
            })
            tool_messages.append(ToolMessage(content=output, tool_call_id=call_id))

        ai_message = AIMessage(
            content=STATE_INJECTION_PLACEHOLDER_TEXT,
            tool_calls=tool_calls,
        )
        return [ai_message, *tool_messages]

    async def _evaluate_statetool_condition(
        self,
        condition: Callable,
        tool_name: str,
        tool_fn: Callable,
    ) -> bool:
        """Run a statetool condition predicate with DI, return a boolean."""
        import typing
        from .tool_utils import INJECTION_MAPPING

        injection_context = {
            "agent": self,
            "tool_name": tool_name,
            "raw_tool": tool_fn,
            "loop_controller": None,
            "react_context": None,
        }
        # Resolve stringified annotations (PEP 563 / `from __future__ import
        # annotations`) so annotation identity comparisons against
        # INJECTION_MAPPING work. Fall back to raw annotations if the
        # whole-function resolve fails (e.g. TYPE_CHECKING-guarded imports).
        try:
            resolved_hints = typing.get_type_hints(condition, include_extras=True)
        except Exception:
            resolved_hints = {}
        args = []
        kwargs = {}
        try:
            sig = inspect.signature(condition)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    target = condition if hasattr(condition, "__self__") else tool_fn
                    self_var = getattr(target, "__self__", None)
                    if self_var:
                        args.insert(0, self_var)
                        continue
                    else:
                        raise ExecutionError("statetool condition error: self referenced on tool that is not also a method")
                inj_type = resolved_hints.get(param_name, param.annotation)
                if inj_type in INJECTION_MAPPING:
                    kwargs[param_name] = injection_context.get(
                        INJECTION_MAPPING[inj_type]
                    )

        result = await ensure_async(condition(*args, **kwargs))
        return bool(result)

    async def _run_statetool(
        self,
        tool_fn: Callable,
        tool_name: str,
    ) -> str:
        """Run a statetool body via its `.run()` method with DI; return a string."""
        tool_context = {
            "agent": self,
            "tool_name": tool_name,
            "raw_tool": tool_fn,
            "loop_controller": None,
            "react_context": None,
        }
        self_ref = getattr(tool_fn, "__self__", None)
        result = await tool_fn.run(
            args={},
            tool_context=tool_context,
            self_ref=self_ref,
        )
        if isinstance(result, str):
            return result
        # Multimodal / non-string returns fall back to repr for now.
        return str(result)

    async def execute(
        self,
        additional_messages: list[BaseMessage] = None,
        tools=None,
        auto_append_response: bool = True
    ) -> AgentResponse:
        if additional_messages is None:
            additional_messages = []
        with self.spinner():
            records = await self.chat_history.records(auto_update_context=True)
            messages = [record.message for record in records] + additional_messages
            tail_messages = await self.build_tail_injections()
            messages.extend(tail_messages)
            self.chat_history.last_sent_messages = list(messages)
            # Communicate the cacheable-prefix boundary to provider
            # preprocessors (plan §4.4). Anthropic / Bedrock read this to
            # attach cache_control to the last persisted message.
            self.model.tail_injection_count = len(tail_messages)
            # TODO: Keep this here?
            token_estimate = await self.chat_history.token_estimate(model=self.model, tools=tools)
            print("Token estimate for query: ", token_estimate)
            if self.verbose:
                self.debug(event_type="llm_request", content=messages)
            raw_result = await self.model.ainvoke(
                input=messages,
                temperature=self.temperature,
                agent_tools=tools,
            )
            usage_metadata = getattr(raw_result, "usage_metadata", None)
            print("Actual usage for query: ", usage_metadata)

        response_token_count = await self.model.token_estimate(messages=[HumanMessage(content=raw_result.content)])

        # Add the raw result to history
        if auto_append_response:
            self.chat_history.add_message(self.model._rectify_result(raw_result), token_count=response_token_count)

        if isinstance(usage_metadata, dict):
            total_token_count = usage_metadata.get("total_tokens", None)
            if total_token_count > token_estimate:
                self.chat_history.base_tokens = total_token_count - token_estimate
        else:
            total_token_count = token_estimate + response_token_count

        # Return processed result
        result = self.model.process_result(raw_result)

        if self.verbose:
            self.debug(event_type="llm_response", content=result)

        def task_callback(task):
            self.post_execute_task = None
        if self.post_execute_task is None:
            task = asyncio.create_task(self.post_execute())
        task.add_done_callback(task_callback)
        self.post_execute_task = task

        return result

    async def oneshot(self, prompt: str, query: str, tools=None) -> str:
        """
        Send a user query to the agent. Returns the agent's response.
        This method ignores any previous conversation history, as well as the existing prompt.
        The output is the raw LLM text withouth any postprocessing, so you'll need to handle parsing it yourself.

        Args:
            prompt (str): The prompt to use when starting a new conversation.
            query (str): The user query to send to the agent.

        Returns:
            str: The agent's response to the user query.
        """
        if tools is None:
            tools = []
        with self.spinner():
            if self.verbose:
                self.debug(event_type="llm_oneshot", content=prompt)
            completion = await self.model.model.ainvoke(
                input=[
                    SystemMessage(content=prompt),
                    HumanMessage(content=query),
                ],
                temperature=self.temperature,
                tools=tools,
            )

        # return the agent's response
        result = completion.content
        if self.verbose:
            self.debug(event_type="llm_response", content=result)
        return result

    def all_messages_sync(self) -> list[BaseMessage]:
        """Synchronous wrapper around the asynchronous all_messages method."""
        return asyncio.run(self.chat_history.all_messages())

    def query_sync(self, message: str) -> str:
        """Synchronous wrapper around the asynchronous query method."""
        return asyncio.run(self.query(message))

    def observe_sync(self, observation: str) -> str:
        """Synchronous wrapper around the asynchronous observe method."""
        return asyncio.run(self.observe(observation))

    def inspect_sync(self, message: str) -> str:
        """Synchronous wrapper around the asynchronous inspect method."""
        return asyncio.run(self.inspect(message))

    def error_sync(self, error: str, drop_error: bool = True) -> str:
        """Synchronous wrapper around the asynchronous error method."""
        return asyncio.run(self.error(error, drop_error))

    def execute_sync(self) -> str:
        """Synchronous wrapper around the asynchronous execute method."""
        return asyncio.run(self.execute())

    def oneshot_sync(self, prompt: str, query: str) -> str:
        """Synchronous wrapper around the asynchronous oneshot method."""
        return asyncio.run(self.oneshot(prompt, query))
