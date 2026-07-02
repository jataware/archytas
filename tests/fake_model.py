"""
A minimal offline model implementation for unit tests that must not hit a
provider API. Token counts are estimated as ``len(content) // 4``.
"""
from types import SimpleNamespace
from typing import Optional, Sequence

from archytas.models.base import BaseArchytasModel, ModelConfig


class FakeChatModel:
    """Stands in for the underlying LangChain chat model."""

    def __init__(self, response_text: str = "SUMMARY TEXT"):
        self.response_text = response_text
        self.calls: list = []

    async def ainvoke(self, input=None, config=None, **kwargs):
        self.calls.append(input)
        return SimpleNamespace(
            content=self.response_text,
            usage_metadata={"input_tokens": 0, "output_tokens": 10, "total_tokens": 10},
        )


class FakeModel(BaseArchytasModel):
    DEFAULT_MODEL = "fake-model"

    def __init__(self, context_window: int = 10000, **kwargs):
        self._context_window = context_window
        super().__init__(ModelConfig(model_name="fake-model"), **kwargs)

    def initialize_model(self, **kwargs):
        return FakeChatModel()

    def contextsize(self, model_name: Optional[str] = None) -> int | None:
        return self._context_window

    async def get_num_tokens_from_messages(
        self,
        messages,
        tools: Optional[Sequence] = None,
    ) -> int:
        total = 0
        for message in messages:
            content = message.content
            if not isinstance(content, str):
                content = str(content)
            total += max(len(content) // 4, 1)
        return total
