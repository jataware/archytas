import re
from langchain_openai.chat_models import ChatOpenAI

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig, set_env_auth


class OpenAIModel(BaseArchytasModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if auth_token:
            set_env_auth(OPENAI_API_KEY=auth_token)
        else:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatOpenAI(model=self.config.get("model_name", "gpt-4o"))

    def _preprocess_messages(self, messages):
        from ..agent import FunctionMessage
        output = []
        for message in messages:
            if isinstance(message, FunctionMessage):
                # Function names in OpenAI messages must match the pattern '^[a-zA-Z0-9_-]+$'
                message.name = re.sub(r'[^a-zA-Z0-9_-]', '_', message.name)
            output.append(message)
        return output
