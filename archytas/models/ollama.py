from langchain_ollama import ChatOllama

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig


class OllamaModel(BaseArchytasModel):

    def auth(self, **kwargs) -> None:
        return

    def initialize_model(self, **kwargs):
        return ChatOllama(model=self.config.get("model_name", "lamma3")) # , api_key=self.api_key)

    def _preprocess_messages(self, messages):
        from ..agent import AgentMessage, SystemMessage, AutoContextMessage
        output = []
    #     system_messages = []
        for message in messages:
            if isinstance(message, AgentMessage):
                import json
                try:
                    json.loads(message.content)
                except Exception as e:
                    print("Got error {e}")
                    raise
                message.content = message.content.strip()
            output.append(message)
    #     output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output
