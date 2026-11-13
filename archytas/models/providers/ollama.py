from langchain_ollama import ChatOllama

from .base import ModelProvider


class OllamaProvider(ModelProvider):
    def initialize_model(self, model, **kwargs):
        # Dummy tool in required to enable "tool mode" on the model
        return ChatOllama(model=model.config.model_name or model.DEFAULT_MODEL)

    def prepare_invoke_kwargs(self, model, kwargs: dict) -> dict:
        if "temperature" in kwargs:
            options = dict(kwargs.get("options") or {})
            options["temperature"] = kwargs.pop("temperature")
            kwargs["options"] = options
        return kwargs

    def contextsize(self, model, model_name: str | None = None) -> int | None:
        if model_name is None:
            model_name = model.model_name
        show_response = model.model._client.show(model.model_name)
        model_info = show_response.modelinfo
        try:
            model_arch = model_info["general.architecture"]
            return int(model_info[f"{model_arch}.context_length"])
        except KeyError:
            print("No context length in model info")
            return None

