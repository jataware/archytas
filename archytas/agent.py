from __future__ import annotations # enable 3.9 support
import os
import openai
import logging
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError, InvalidRequestError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type as retry_if, stop_after_attempt, wait_exponential
from typing import TypedDict, Literal


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

class Agent:
    def __init__(self, *, model:str='gpt-4', prompt:str="You are a helpful assistant.", api_key:str|None=None):
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

        # check that an api key was given, and set it
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', None)
        if not api_key:
            raise Exception("No OpenAI API key given. Please set the OPENAI_API_KEY environment variable or pass the api_key argument to the Agent constructor.")
        openai.api_key = api_key


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

        # Drop error + original bad input from chat history
        if drop_error:
            del self.messages[-3:-1]

        return result
    
    @retry
    def execute(self) -> str:
        with Live(Spinner('dots', speed=2, text="thinking..."), refresh_per_second=30, transient=True):
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model, 
                    messages=[self.system_message] + self.messages,
                    temperature=0,
                )
            except InvalidRequestError as e:
                print(self.messages)
                import pdb;pdb.set_trace()
                ...
        
        # grab the response and add it to the chat history
        result = completion.choices[0].message.content
        self.messages.append({"role": Role.assistant, "content": result})

        return result
