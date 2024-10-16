from langchain_openai.chat_models import ChatOpenAI

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig


class OpenAIAuth(EnvironmentAuth):
    def __init__(self, api_key: str) -> None:
        super().__init__(OPENAI_API_KEY=api_key)

class OpenAIModel(BaseArchytasModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def auth(self, **kwargs) -> None:
        auth_object = None
        if 'api_key' in kwargs:
            auth_object = OpenAIAuth(kwargs['api_key'])
        elif 'api_key' in self.config:
            auth_object = OpenAIAuth(self.config['api_key'])
        if auth_object:
            auth_object.apply()
        else:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatOpenAI(model=self.config.get("model_name", "gpt-4o"))

    def _preprocess_messages(self, messages):
        from ..agent import AgentMessage, SystemMessage
        output = []
        for message in messages:
            if isinstance(message, AgentMessage):
                output.append(SystemMessage(message.content))
            else:
                output.append(message)
        return output
