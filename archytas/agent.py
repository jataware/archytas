import logging
import os
import toml
import openai
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
    def __init__(self, *, model:str='gpt-4', prompt:str="You are a helpful assistant."):
        self.model = model
        self.system_message: Message = {"role": Role.system, "content": prompt }
        self.messages = []

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
