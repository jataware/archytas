import logging
import openai
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError, InvalidRequestError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type as retry_if, stop_after_attempt, wait_exponential
from enum import Enum
from typing import TypedDict, Literal

logger = logging.getLogger(__name__)
# retry = tenacity_retry(
#     reraise=True,
#     stop=stop_after_attempt(4),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if(Timeout) | retry_if(APIError) | retry_if(APIConnectionError) | retry_if(RateLimitError) | retry_if(ServiceUnavailableError),
#     before_sleep=before_sleep_log(logger, logging.WARNING),
# )

from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
retry_decorator = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
          retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)
        | retry_if_exception_type(openai.error.ServiceUnavailableError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

class Role:
    system = 'system'
    assistant = 'assistant'
    user = 'user'
class Message(TypedDict):
    role: Literal['system', 'assistant', 'user']
    content: str

class Agent:
    def __init__(self, model:str='gpt-3.5-turbo', prompt:str="You are a helpful assistant."):
        self.model = model
        self.system_message: Message = {"role": Role.system, "content": prompt }
        self.messages = []


    def query(self, message:str) -> str:
        return self(message, role=Role.user)
    
    def observe(self, observation:str) -> str:
        return self(observation, role=Role.system)
    
    def error(self, error:str) -> str:
        return self(f"ERROR: {error}", role=Role.system)
    
    def __call__(self, message:str, role:Literal['system', 'user']=Role.user):
        self.messages.append({"role": role, "content": message})
        result = self.execute()
        self.messages.append({"role": Role.assistant, "content": result})
        
        # Drop error + correction from chat history
        if message.startswith('ERROR:'):
            del self.messages[-3:-1]

        return result
    
    @retry_decorator
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
