import logging
import openai
from openai.error import Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError, InvalidRequestError
from tenacity import before_sleep_log, retry as tenacity_retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)
retry = tenacity_retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Timeout, APIError, APIConnectionError, RateLimitError, ServiceUnavailableError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class Agent:
    def __init__(self, model:str='gpt-3.5-turbo', system:str="You are a helpful assistant.", role:str="assistant"):
        self.model = model
        self.role = role
        self.system_message = {"role": "system", "content": system}
        self.messages = []
    
    def __call__(self, message:str):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": self.role, "content": result})
        
        # Drop error + correction from chat history
        if message.startswith('ERROR:'):
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
        except InvalidRequestError:
            print(self.messages)
            breakpoint()
        
        return completion.choices[0].message.content
