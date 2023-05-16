import os
import openai
import logging
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type as retry_if, stop_after_attempt, wait_exponential
from typing import Literal, Callable, ContextManager
from enum import Enum

from rich.spinner import Spinner
from rich.live import Live


logger = logging.getLogger(__name__)
retry = tenacity_retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if((Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

class Role(str, Enum):
    system = 'system'
    assistant = 'assistant'
    user = 'user'

class Message(dict):
    """Message format for communicating with the OpenAI API."""
    def __init__(self, role:Role, content:str):
        super().__init__(role=role.value, content=content)

class ContextMessage(Message):
    """Simple wrapper around a message that adds an id and optional lifetime."""
    def __init__(self, role:Role, content:str, id:int, lifetime:int|None=None):
        super().__init__(role=role, content=content)
        self.id = id
        self.lifetime = lifetime


def cli_spinner(): 
    return Live(Spinner('dots', speed=2, text="thinking..."), refresh_per_second=30, transient=True)
class no_spinner:
    def __enter__(self): pass
    def __exit__(self, *args): pass

class Agent:
    def __init__(self, *, model:str='gpt-4', prompt:str="You are a helpful assistant.", api_key:str|None=None, spinner:Callable[[], ContextManager]|None=cli_spinner):
        """
        Agent class for managing communication with OpenAI's API.

        Args:
            model (str, optional): The name of the model to use. Defaults to 'gpt-4'. At present, GPT-4 is the only model that works reliably.
            prompt (str, optional): The prompt to use when starting a new conversation. Defaults to "You are a helpful assistant.".
            api_key (str, optional): The OpenAI API key to use. Defaults to None. If None, the API key will be read from the OPENAI_API_KEY environment variable.

        Raises:
            Exception: If no API key is given.
        """

        self.model = model
        self.system_message = Message(role=Role.system, content=prompt)
        self.messages: list[Message] = []
        self.spinner = spinner if spinner is not None else no_spinner

        # use to generate unique ids for context messages
        self._current_context_id = 0

        # check that an api key was given, and set it
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', None)
        if not api_key:
            raise Exception("No OpenAI API key given. Please set the OPENAI_API_KEY environment variable or pass the api_key argument to the Agent constructor.")
        openai.api_key = api_key

    def new_context_id(self) -> int:
        """Generate a new context id."""
        self._current_context_id += 1
        return self._current_context_id

    def add_context(self, context:str, *, lifetime:int|None=None) -> int:
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
        context_message = ContextMessage(role=Role.system, content=context, id=self.new_context_id(), lifetime=lifetime)
        self.messages.append(context_message)
        return context_message.id

    def update_timed_context(self) -> None:
        """
        Update the lifetimes of all timed contexts, and remove any that have expired.
        This should be called after every LLM response.
        """
        #decrement lifetimes of all timed context messages
        for message in self.messages:
            if isinstance(message, ContextMessage) and message.lifetime is not None:
                message.lifetime -= 1

        #remove expired context messages
        new_messages = []
        for message in self.messages:
            if isinstance(message, ContextMessage) and message.lifetime == 0:
                continue
            new_messages.append(message)
        self.messages = new_messages


    def clear_context(self, id:int) -> None:
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
        self.messages = [message for message in self.messages if not isinstance(message, ContextMessage)]
    
    def query(self, message:str) -> str:
        """Send a user query to the agent. Returns the agent's response"""
        self.messages.append(Message(role=Role.user, content=message))
        return self.execute()
    
    def observe(self, observation:str) -> str:
        """Send a system/tool observation to the agent. Returns the agent's response"""
        self.messages.append(Message(role=Role.system, content=observation))
        return self.execute()
    
    def error(self, error:str, drop_error:bool=True) -> str:
        """
        Send an error message to the agent. Returns the agent's response.
        
        Args:
            error (str): The error message to send to the agent.
            drop_error (bool, optional): If True, the error message and LLMs bad input will be dropped from the chat history. Defaults to `True`.
        """
        self.messages.append(Message(role=Role.system, content=f'ERROR: {error}'))
        result = self.execute()

        # Drop error + LLM's bad input from chat history
        if drop_error:
            del self.messages[-3:-1]

        return result
    
    @retry
    def execute(self) -> str:
        with self.spinner():
            completion = openai.ChatCompletion.create(
                model=self.model, 
                messages=[self.system_message] + self.messages,
                temperature=0,
            )
        
        # grab the response and add it to the chat history
        result = completion.choices[0].message.content
        self.messages.append(Message(role=Role.assistant, content=result))

        # remove any timed contexts that have expired
        self.update_timed_context()

        return result

    @retry
    def oneshot(self, prompt:str, query:str) -> str:
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
        with self.spinner():
            completion = openai.ChatCompletion.create(
                model=self.model, 
                messages=[Message(role=Role.system, content=prompt), Message(role=Role.user, content=query)],
                temperature=0,
            )

        # return the agent's response
        result = completion.choices[0].message.content
        return result
