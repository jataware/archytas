import asyncio
import inspect
import json
import logging
import os
import warnings
from enum import Enum
from tenacity import (
    before_sleep_log,
    retry as tenacity_retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from typing import Callable, ContextManager, Any, Optional, Literal
from openai import (
    APITimeoutError,
    APIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)
from enum import Enum
from functools import wraps
from typing import Callable, ContextManager, Any
from pydantic import BaseModel as PydanticModel

from rich import print as rprint
from rich.spinner import Spinner
from rich.live import Live


from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage

from .models.base import BaseArchytasModel
from .models.openai import OpenAIModel


logger = logging.getLogger(__name__)


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


class AuthenticationError(Exception):
    pass
from langchain_anthropic import ChatAnthropic
# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser



logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    pass


class AgentMessage(BaseMessage):
    """
    Agent Message type that plays well with langchain.
    """
    type: Literal["agent"] = "agent"


class Role(str, Enum):
    system = "system"
    agent = "agent"
    user = "user"


class ContextMessage(SystemMessage):
    """Simple wrapper around a message that adds an id and optional lifetime."""

    id: int
    lifetime: int | None = None


class AutoContextMessage(SystemMessage):
    """An automatically updating context message that remains towards the top of the message list."""

    default_content: str
    content_updater: Callable[[], str]

    def __init__(self, default_content: str, content_updater: Callable[[], str], **kwargs):
        super().__init__(content=default_content, default_content=default_content, content_updater=content_updater, **kwargs)

    async def update_content(self):
        if inspect.iscoroutinefunction(self.content_updater):
            result = await self.content_updater()
        else:
            result = self.content_updater()
        self.content = result


def cli_spinner():
    return Live(
        Spinner("dots", speed=2, text="thinking..."),
        refresh_per_second=30,
        transient=True,
    )


def auth(func):
    """
    """
    @wraps(func)
    async def inner(*args, **kwargs):
        # if not openai.api_key:
        #     raise AuthenticationError(
        #         "No OpenAI API key given. Please set the OPENAI_API_KEY environment variable or pass the api_key argument to the Agent constructor."
        #     )
        import openai
        try:
            return await func(*args, **kwargs)
        except (openai.AuthenticationError, openai.OpenAIError) as err:
            raise AuthenticationError(err)
    return inner



class no_spinner:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Agent:
    def __init__(
        self,
        *,
        model: BaseArchytasModel,
        prompt: str = "You are a helpful assistant.",
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
            prompt (str, optional): The prompt to use when starting a new conversation. Defaults to "You are a helpful assistant.".
            api_key (str, optional): The OpenAI API key to use. Defaults to None. If None, the API key will be read from the OPENAI_API_KEY environment variable.
            spinner ((fn -> ContextManager) | None, optional): A function that returns a context manager that is run every time the LLM is generating a response. Defaults to cli_spinner which is used to display a spinner in the terminal.
            rich_print (bool, optional): Whether to use rich to print messages. Defaults to True. Can also be set via the DISABLE_RICH_PRINT environment variable.
            verbose (bool, optional): Expands the debug output. Includes full query context on requests to the LLM. Defaults to False.
            messages (list[BaseMessage], optional): A list of messages to initialize the agent's conversation history with. Defaults to an empty list.

        Raises:
            Exception: If no API key is given.
        """
        # import openai

        self.rich_print = bool(
            rich_print and not os.environ.get("DISABLE_RICH_PRINT", False)
        )
        self.verbose = verbose
        if model is None:
            self.model = OpenAIModel({"api_key": api_key})
        else:
            self.model = model
        self.system_message = SystemMessage(content=prompt)
        self.messages: list[BaseMessage] = []
        if spinner is not None and self.rich_print:
            self.spinner = spinner
        else:
            self.spinner = no_spinner

        # use to generate unique ids for context messages
        self._current_context_id = 0

        # Initialize the auto_context_message to empty
        self.auto_context_message = None
        self.auto_update_context = False

        # check that an api key was given, and set it
        # if api_key is None:
        #     api_key = os.environ.get("OPENAI_API_KEY", None)
        # openai.api_key = api_key
        self.temperature = temperature

    def print(self, *args, **kwargs):
        if self.rich_print:
            rprint(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def debug(self, event_type: str, content: Any) -> None:
        """
        Debug handler

        Function at this level so it can be overridden in a subclass.
        """
        logger.debug(event_type, content)

    def new_context_id(self) -> int:
        """Generate a new context id."""
        self._current_context_id += 1
        return self._current_context_id

    async def all_messages(self) -> list[BaseMessage]:
        messages = [self.system_message]
        if self.auto_context_message:
            if self.auto_update_context:
                await self.auto_context_message.update_content()
            messages.append(self.auto_context_message)
        messages.extend(self.messages)
        return messages

    def add_context(self, context: str, *, lifetime: int | None = None) -> int:
        """
        Inject a context message to the agent's conversation.

        Useful for providing the agent with information relevant to the current conversation, e.g. tool state, environment info, etc.
        If a lifetime is specified, the context message will automatically be deleted from the chat history after that many steps.
        A context message can be deleted manually by calling clear_context() with the id of the context message.

        Args:
            context (str): The context to add to the agent's conversation.
            lifetime (int, optional): The number of time steps the context will live for. Defaults to None (i.e. it will never be removed).

        Returns:
            int: The id of the context message.
        """
        context_message = ContextMessage(
            content=context,
            id=self.new_context_id(),
            lifetime=lifetime,
        )
        self.messages.append(context_message)
        return context_message.id

    def update_timed_context(self) -> None:
        """
        Update the lifetimes of all timed contexts, and remove any that have expired.
        This should be called after every LLM response.
        """
        # decrement lifetimes of all timed context messages
        for message in self.messages:
            if isinstance(message, ContextMessage) and message.lifetime is not None:
                message.lifetime -= 1

        # remove expired context messages
        new_messages = []
        for message in self.messages:
            if isinstance(message, ContextMessage) and message.lifetime == 0:
                continue
            new_messages.append(message)
        self.messages = new_messages

    def clear_context(self, id: int) -> None:
        """
        Remove a single context message from the agent's conversation.

        Args:
            id (int): The id of the context message to remove.
        """
        new_messages = []
        for message in self.messages:
            if isinstance(message, ContextMessage) and message.id == id:
                continue
            new_messages.append(message)
        self.messages = new_messages

    def clear_all_context(self) -> None:
        """Remove all context messages from the agent's conversation."""
        self.messages = [
            message
            for message in self.messages
            if not isinstance(message, ContextMessage)
        ]

    def set_auto_context(
        self,
        default_content: str,
        content_updater: Callable[[], str] | None = None,
        auto_update: bool = True,
    ):
        """
        A special type of context message that is always towards the top (but after the prompt and any system messages).
        This allows an agent to automatically update its context based on live conditions without having to call a tool every time.

        Args:
            default_content (str): The default message/content of the context if the content updater has not or cannot be run.
            content_updater (callable): A function/lambda that takes no arguments and returns a string. The returned string
                                        becomes the new context value.
            auto_update (boolean): If true, the context will be updated on every call. Otherwise, the context can be updated by
                                   calling `agent.auto_context_message.update_content()` when desired.
        """
        self.auto_update_context = auto_update
        if not self.auto_context_message:
            self.auto_context_message = AutoContextMessage(
                default_content=default_content,
                content_updater=content_updater,
            )
        else:
            self.auto_context_message.update(
                default_content=default_content,
                content_updater=content_updater,
            )

    async def handle_message(self, message: BaseMessage):
        """Appends a message to the message list and executes."""
        self.messages.append(message)
        return await self.execute()

    async def query(self, message: str) -> str:
        """Send a user query to the agent. Returns the agent's response"""
        return await self.handle_message(HumanMessage(content=message))

    async def observe(self, observation: str) -> str:
        """Send a system/tool observation to the agent. Returns the agent's response"""
        return await self.handle_message(AIMessage(content=observation))

    async def inspect(self, query: str) -> str:
        """Send one-off system query that is not recorded in history"""
        return await self.execute([HumanMessage(content=query)])

    async def error(self, error: str, drop_error: bool = True) -> str:
        """
        Send an error message to the agent. Returns the agent's response.

        Args:
            error (str): The error message to send to the agent.
            drop_error (bool, optional): If True, the error message and LLMs bad input will be dropped from the chat history. Defaults to `True`.
        """
        result = await self.handle_message(HumanMessage(content=f"ERROR: {error}"))

        # Drop error + LLM's bad input from chat history
        if drop_error:
            del self.messages[-3:-1]

        return result

    @auth
    async def execute(self, additional_messages: list[BaseMessage] = []) -> str:
        with self.spinner():
            messages = (await self.all_messages()) + additional_messages
            if self.verbose:
                self.debug(event_type="llm_request", content=messages)
            raw_result = await self.model.ainvoke(
                input=messages,
                temperature=self.temperature,
            )

        print('post-execute', type(raw_result), raw_result)

        # Add the raw result to history
        match raw_result:
            case BaseMessage():
                self.messages.append(raw_result)
            case PydanticModel():
                self.messages.append(AIMessage(content=json.dumps(raw_result.model_dump())))
            case dict():
                self.messages.append(AIMessage(content=json.dumps(raw_result)))
            case _:
                raise TypeError(f"Unable to handle agent raw_result of type {type(raw_result)}. {str(raw_result)[:100]}...")


        # Return processed result
        result = self.model.process_result(raw_result)
        print(type(result), result)

        if self.verbose:
            self.debug(event_type="llm_response", content=result)

        # if isinstance(result, BaseMessage):
        #     self.messages.append(result)
        # elif isinstance(result, str):
        #     self.messages.append(AIMessage(content=result))
        # elif isinstance(result, dict) and "content" in result:
        #     self.messages.append(AIMessage(**result))
        # elif isinstance(result, dict):
        #     self.messages.append(AIMessage(content=result))
        # else:
        #     raise TypeError(f"Unable to handle agent result of type {type(result)}. {str(result)[:100]}...")
        # if self.verbose:
        #     self.debug(event_type="llm_response", content=result)

        # remove any timed contexts that have expired
        self.update_timed_context()

        return result

    @auth
    async def oneshot(self, prompt: str, query: str) -> str:
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
        import openai
        with self.spinner():
            if self.verbose:
                self.debug(event_type="llm_oneshot", content=prompt)
            completion = self.model.invoke(
                input=[
                    SystemMessage(content=prompt),
                    HumanMessage(content=query),
                ],
                temperature=self.temperature,
            )

        # return the agent's response
        result = completion.content
        if self.verbose:
            self.debug(event_type="llm_response", content=result)
        return result

    def all_messages_sync(self) -> list[BaseMessage]:
        """Synchronous wrapper around the asynchronous all_messages method."""
        return asyncio.run(self.all_messages())

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
