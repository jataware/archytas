from langchain_anthropic import ChatAnthropic

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig


class AnthropicAuth(EnvironmentAuth):
    def __init__(self, api_key: str) -> None:
        super().__init__(ANTHROPIC_API_KEY=api_key)

class AnthropicModel(BaseArchytasModel):
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
        return ChatAnthropic(model=self.config.get("model_name", "claude-2.1"), api_key=self.api_key)

    def _preprocess_messages(self, messages):
        from ..agent import AgentMessage, SystemMessage, AutoContextMessage
        output = []
        system_messages = []
        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            elif isinstance(message, AgentMessage):
                output[-1].content += f"\nRunning the tool above returned the following output:\n```\n{message.content}\n```"
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output
