from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig, set_env_auth


class GeminiModel(BaseArchytasModel):
    api_key: str

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if auth_token:
            self.api_key = auth_token
        else:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatGoogleGenerativeAI(model=self.config.get("model_name", "gpt-4o"), api_key=self.api_key)

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        kwargs.pop("temperature")
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    def _preprocess_messages(self, messages):
        from ..agent import AgentMessage, SystemMessage, AutoContextMessage, AIMessage
        output = []
        system_messages = []
        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output
