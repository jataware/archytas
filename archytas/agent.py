import os
import openai
import logging
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError, InvalidRequestError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type as retry_if, stop_after_attempt, wait_exponential
from typing import TypedDict, Literal, Callable, ContextManager
from frozendict import frozendict

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

#TODO: want these to be enums, but Role.role needs to return a string, not an enum member
class Role:
    system = 'system'
    assistant = 'assistant'
    user = 'user'
class Message(TypedDict):
    role: Literal['system', 'assistant', 'user']
    content: str


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
        self.system_message: Message = {"role": Role.system, "content": prompt }
        self.messages = []
        self.spinner = spinner if spinner is not None else no_spinner

        # keep track of injected context messages and their lifetimes
        self._context_lifetimes = {}
        self._all_context_messages = []

        # check that an api key was given, and set it
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', None)
        if not api_key:
            raise Exception("No OpenAI API key given. Please set the OPENAI_API_KEY environment variable or pass the api_key argument to the Agent constructor.")
        openai.api_key = api_key


    def add_timed_context(self, context:str, time:int=1) -> None:
        """
        Add a context to the agent's conversation.
        The context will be added to the conversation for a finite number of time steps.

        Args:
            context (str): The context to add to the agent's conversation.
            time (int, optional): The number of time steps the context will live for. Defaults to 1 (i.e. it gets deleted as soon as the LLM sees it once).
        """
        context_message = frozendict({"role": Role.system, "content": context})
        self.messages.append(context_message)
        self._context_lifetimes[context_message] = time
        self._all_context_messages.append(context_message)


    def update_timed_context(self) -> None:
        """
        Update the lifetimes of all timed contexts, and remove any that have expired.
        This should be called after every LLM response.
        """
        # Update context lifetimes and remove expired contexts
        for context_message in list(self._context_lifetimes.keys()):
            self._context_lifetimes[context_message] -= 1
            if self._context_lifetimes[context_message] <= 0:
                self.messages.remove(context_message)
                self._all_context_messages.remove(context_message)
                del self._context_lifetimes[context_message]

    
    def add_permanent_context(self, context:str) -> None:
        """
        Add a context to the agent's conversation.
        The context will be added to the conversation permanently.

        Args:
            context (str): The context to add to the agent's conversation.
        """
        context_message = {"role": Role.system, "content": context}
        self.messages.append(context_message)
        self._all_context_messages.append(context_message)


    def add_managed_context(self, context:str) -> Callable[[], None]:
        """
        Add a context to the agent's conversation.
        The context will be added to the conversation for an arbitrary number of time steps.
        Returns a function that can be called to remove the context from the conversation.

        Args:
            context (str): The context to add to the agent's conversation.

        Returns:
            Callable[[], None]: A function that can be called to remove the context from the conversation.
        """
        context_message = {"role": Role.system, "content": context}
        self.messages.append(context_message)
        self._all_context_messages.append(context_message)

        def remove_context():
            self.messages.remove(context_message)
            self._all_context_messages.remove(context_message)

        return remove_context

    def clear_all_context(self) -> None:
        """Remove all contexts from the agent's conversation."""
        for context_message in self._all_context_messages:
            self.messages.remove(context_message)
        self._all_context_messages = []
        self._context_lifetimes = {}
    
    def query(self, message:str) -> str:
        """Send a user query to the agent. Returns the agent's response"""
        self.messages.append({"role": Role.user, "content": message})
        return self.execute()
    
    def observe(self, observation:str) -> str:
        """Send a system/tool observation to the agent. Returns the agent's response"""
        self.messages.append({"role": Role.system, "content": observation})
        return self.execute()
    
    def error(self, error:str, drop_error:bool=True) -> str:
        """
        Send an error message to the agent. Returns the agent's response.
        
        Args:
            error (str): The error message to send to the agent.
            drop_error (bool, optional): If True, the error message and LLMs bad input will be dropped from the chat history. Defaults to `True`.
        """
        self.messages.append({"role": Role.system, "content": f'ERROR: {error}'})
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
        self.messages.append({"role": Role.assistant, "content": result})

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
                messages=[{"role": Role.system, "content": prompt }, {"role": Role.user, "content": query}],
                temperature=0,
            )

        # return the agent's response
        result = completion.choices[0].message.content
        return result
