import json
from langchain_groq import ChatGroq

from archytas.agent import AIMessage, HumanMessage, FunctionMessage

from .base import BaseArchytasModel, set_env_auth, ModelConfig


class GroqModel(BaseArchytasModel):
    api_key: str = ""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        elif 'api_key' in self.config:
            self.api_key = self.config['api_key']
        if not self.api_key:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatGroq(model=self.config.get("model_name", "llama3-8b-8192"), api_key=self.api_key)

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    # def _preprocess_messages(self, messages):
    #     from ..agent import AgentMessage, SystemMessage, AutoContextMessage, ToolMessage, AIMessage
    #     output = []
    #     for message in messages:
    #         else:
    #             output.append(message)
    #     print("\n----\n".join(map(str, output)))
    #     print('=================')
    #     return output

    # def process_result(self, result_message: AIMessage):
    #     return result_message.get("content")
