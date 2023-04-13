import logging
import openai
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError, InvalidRequestError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type as retry_if, stop_after_attempt, wait_exponential
from enum import Enum
from typing import TypedDict, Literal

logger = logging.getLogger(__name__)
retry = tenacity_retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if(Timeout) | retry_if(APIError) | retry_if(APIConnectionError) | retry_if(RateLimitError) | retry_if(ServiceUnavailableError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
# retry_decorator = retry(
#     reraise=True,
#     stop=stop_after_attempt(4),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=(
#           retry_if_exception_type(openai.error.Timeout)
#         | retry_if_exception_type(openai.error.APIError)
#         | retry_if_exception_type(openai.error.APIConnectionError)
#         | retry_if_exception_type(openai.error.RateLimitError)
#         | retry_if_exception_type(openai.error.ServiceUnavailableError)
#     ),
#     before_sleep=before_sleep_log(logger, logging.WARNING),
# )

class Role:
    system = 'system'
    assistant = 'assistant'
    user = 'user'
class Message(TypedDict):
    role: Literal['system', 'assistant', 'user']
    content: str

class Agent:
    def __init__(self, model:str='gpt-4', prompt:str="You are a helpful assistant."):
        self.model = model
        self.system_message: Message = {"role": Role.system, "content": prompt }
        self.messages = []

    def query(self, message:str) -> str:
        """Send a user query to the agent. Returns the agent's response"""
        self.messages.append({"role": Role.user, "content": message})
        result = self.execute()
        self.messages.append({"role": Role.assistant, "content": result})
        return result
    
    def observe(self, observation:str) -> str:
        """Send a system/tool observation to the agent. Returns the agent's response"""
        self.messages.append({"role": Role.system, "content": observation})
        result = self.execute()
        self.messages.append({"role": Role.assistant, "content": result})
        return result
    
    def error(self, error:str, drop_error:bool=True) -> str:
        """
        Send an error message to the agent. Returns the agent's response.
        
        `error`: The error message to send to the agent.
        `drop_error`: (optional) If True, the error message and LLMs bad input will be dropped from the chat history. Defaults to `True`.
        """
        self.messages.append({"role": Role.system, "content": f'ERROR: {error}'})
        result = self.execute()
        self.messages.append({"role": Role.assistant, "content": result})

        # Drop error + original bad input from chat history
        if drop_error:
            del self.messages[-3:-1]

        return result
    
    @retry
    def execute(self):
        try:
            completion = openai.ChatCompletion.create(
                model=self.model, 
                messages=[self.system_message] + self.messages,
                temperature=0,
            )
        except InvalidRequestError as e:
            print(self.messages)
            # breakpoint()
            import pdb;pdb.set_trace()
            1
        
        return completion.choices[0].message.content
