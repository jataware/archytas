import pdb
import os
import asyncio
import requests
import json
from typing import Generator, Literal, overload, TypedDict, Any, Optional, Sequence, ClassVar, cast
from pathlib import Path
from functools import lru_cache
import logging

here = Path(__file__).parent


from .openrouter_models import ModelName, attributes_map
from .base import BaseArchytasModel, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, FunctionMessage
Role = Literal["user", "assistant", 'system']

class Message(TypedDict):
    role: Role
    content: str


class Model:
    """A simple class for talking to OpenRouter models directly via requests to the API"""
    def __init__(self, model:ModelName, openrouter_api_key:str):
        self.model = model
        self.openrouter_api_key = openrouter_api_key
    
    @overload
    def complete(self, messages: list[Message], stream:Literal[False]=False) -> str: ...
    @overload
    def complete(self, messages: list[Message], stream:Literal[True]) -> Generator[str, None, None]: ...
    def complete(self, messages: list[Message], stream:bool=False) -> str | Generator[str, None, None]:
        if stream:
            return self._streaming_complete(messages)
        else:
            return self._blocking_complete(messages)


    def _blocking_complete(self, messages: list[Message]) -> str:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            data=json.dumps({"model": self.model, "messages": messages})
        )
        data = response.json()
        try:
            return data['choices'][0]['message']['content']
        except KeyError as e:
            raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
        except Exception as e:
            raise ValueError(f"An error occurred while processing the response: '{data}'. {e}") from e

    # TODO: should request timeout be a setting rather than hardcoded?
    def _streaming_complete(self, messages: list[Message]) -> Generator[str, None, None]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {"model": self.model, "messages": messages, "stream": True}

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            buf = []
            for line in r.iter_lines(decode_unicode=True, chunk_size=1024):
                if line is None:
                    continue
                
                if line.startswith("data: "):
                    buf.append(line[6:])
                    continue

                if line == "":  # end of one SSE event
                    if not buf:
                        continue
                    data = "\n".join(buf)
                    buf.clear()

                    if data == "[DONE]":
                        return

                    # parse the chunk and yield any content
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue # wait for the next complete event
                    try:
                        content = obj["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except KeyError as e:
                        raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
                    
                    continue

                
                # ignore other fields like "event:" / "id:" / comments




def list_openrouter_models(api_key:str) -> list[str]:
    """get the list of model names from OpenRouter API"""
    models = _get_openrouter_models(api_key)
    return [model['id'] for model in models]

@lru_cache()
def _get_openrouter_models(api_key: str) -> list[dict]:
    """Get the list of models (including all metadata) from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}")

    data = response.json()['data']
    return data


# TODO: make a convenient project-level command for calling this function
def create_models_types_file(api_key: str, file:Path=here/'openrouter_models.py'):
    """
    Create a file with a type declaration `ModelName` containing all OpenRouter models.
    """
    models = _get_openrouter_models(api_key)
    models.sort(key=lambda x: x['id'])
    model_names = [model['id'] for model in models]
    context_lengths = [model['context_length'] for model in models]
    supports_tools = ["tools" in model['supported_parameters'] for model in models]


    print(f"Creating {file.relative_to(here.parent)} with {len(models)} models")
    name_lines = ',\n    '.join(f"'{model_name}'" for model_name in model_names)
    attributes_lines = ',\n    '.join(f'''{f'"{model_name}":':<40} Attr(context_size={context_length}, supports_tools={supports_tool})''' for model_name, context_length, supports_tool in zip(model_names, context_lengths, supports_tools))
    file.write_text(f'''\
# DO NOT EDIT THIS FILE MANUALLY
# This file is generated by calling `create_models_types_file()` in archytas/models/openrouter.py

from typing import Literal
from dataclasses import dataclass


ModelName = Literal[
    {name_lines}
]


@dataclass
class Attr:
    context_size: int
    supports_tools: bool
    # TBD: may add more in the future


attributes_map: dict[ModelName, Attr] = {{
    {attributes_lines}
}}
''')



# if __name__ == "__main__":
#     import os
#     create_models_types_file(api_key=os.getenv('OPENROUTER_API_KEY'))


# model['supported_parameters']
# """
# possible parameters: {'response_format', 'max_tokens', 'top_k', 'top_p', 'logprobs', 'temperature', 'structured_outputs', 'top_a', 'min_p', 'include_reasoning', 'reasoning', 'tool_choice', 'stop', 'top_logprobs', 'seed', 'frequency_penalty', 'repetition_penalty', 'tools', 'presence_penalty', 'web_search_options', 'logit_bias'}
# """



# from .model import Model, Message, Role
# from typing import Literal, Generator, overload

class Agent:
    """Basically just a model paired with message history tracking"""
    def __init__(self, model: Model):
        self.model = model
        self.messages: list[Message] = []

    def add_message(self, role: Role, content: str):
        message = Message(role=role, content=content)
        self.messages.append(message)
    
    def add_user_message(self, content: str):
        self.add_message(role='user', content=content)

    def add_assistant_message(self, content: str):
        self.add_message(role='assistant', content=content)

    def add_system_message(self, content: str):
        self.add_message(role='system', content=content)
    
    @overload
    def execute(self, stream:Literal[False]=False) -> str: ...
    @overload
    def execute(self, stream:Literal[True]) -> Generator[str, None, None]: ...
    def execute(self, stream:bool=False) -> str | Generator[str, None, None]:
        if stream:
            return self._streaming_execute()
        else:
            return self._blocking_execute()
    
    def _blocking_execute(self) -> str:
        result = self.model.complete(self.messages, stream=False)
        self.add_assistant_message(result)
        return result

    def _streaming_execute(self) -> Generator[str, None, None]:
        # stream the chunks while also capturing them
        result_chunks = []
        for chunk in self.model.complete(self.messages, stream=True):
            result_chunks.append(chunk)
            yield chunk
        
        # add the message to the history after streaming is done
        self.add_assistant_message(''.join(result_chunks))


# -----------------------
# Archytas provider wrapper
# -----------------------

class ChatOpenRouter:
    """
    Minimal adapter that emulates a chat model interface over OpenRouter's REST API.

    Supports invoke/ainvoke, bind_tools (no-op), and a token estimator.
    """
    def __init__(self, *, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tools: Optional[Sequence[Any]] = None

        # get the model attributes
        try:
            self._attrs = attributes_map[cast(ModelName, model)]
        except KeyError:
            from .openrouter_models import Attr
            self._attrs = Attr(context_size=200_000, supports_tools=True)
            logging.warning(f"Unrecognized OpenRouter model: '{model}' (this implies model is not officially listed in archytas/models/openrouter_models.py). Consider regenerating the file with `create_models_types_file()` to get most up-to-date models list. Attempting to continue with the following Attributes: {self._attrs}")
        
        if not self._attrs.supports_tools:
            raise ValueError(f"OpenRouter model '{model}' does not support tools. Archytas requires models to support tools. Please use a different model.")

    def bind_tools(self, tools: Sequence[Any]):
        self._tools = tools
        self._schemas = []
        for tool in tools:
            langchain_schema: dict = tool.tool_call_schema.schema()
            schema = {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': langchain_schema['properties'],
                    'required': langchain_schema['required'],
                }
            }
            self._schemas.append(schema)
        
        return self

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        def serialize_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(json.dumps(item))
                return "\n".join(parts)
            return json.dumps(content)

        converted: list[dict[str, str]] = []
        for msg in messages:
            match msg:
                case HumanMessage():
                    converted.append({"role": "user", "content": serialize_content(msg.content)})
                case AIMessage():
                    converted.append({"role": "assistant", "content": serialize_content(msg.content)})
                case SystemMessage():
                    converted.append({"role": "system", "content": serialize_content(msg.content)})
                case ToolMessage():
                    converted.append({"role": "system", "content": f"Tool {getattr(msg, 'name', 'output')}: {serialize_content(msg.content)}"})
                case FunctionMessage():
                    converted.append({"role": "system", "content": f"Observation from {getattr(msg, 'name', 'function')}: {serialize_content(msg.content)}"})
                case _:
                    converted.append({"role": "user", "content": serialize_content(msg.content)})
        return converted

    def get_num_tokens_from_messages(self, *, messages: list[BaseMessage], tools: Optional[Sequence[Any]] = None) -> int:
        # Rough heuristic: 4 chars per token
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            else:
                total_chars += len(json.dumps(msg.content))
        return max(1, total_chars // 4)

    def invoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(input),
        }
        # Pass through common generation params if present
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.base_url, headers=headers, json=payload)
        if resp.status_code == 401:
            raise AuthenticationError("OpenRouter Authentication Error")
        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {"error": resp.text}
            raise ExecutionError(str(data))

        data = resp.json()
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except Exception as err:
            raise ExecutionError(f"Unexpected response from OpenRouter: {data}") from err

        usage = data.get("usage")
        usage_metadata = None
        if isinstance(usage, dict):
            # Only include ints; omit missing/unknowns to satisfy typing
            meta: dict[str, int] = {}
            if isinstance(usage.get("prompt_tokens"), int):
                meta["input_tokens"] = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                meta["output_tokens"] = usage["completion_tokens"]
            if isinstance(usage.get("total_tokens"), int):
                meta["total_tokens"] = usage["total_tokens"]
            if meta:
                usage_metadata = meta

        message = AIMessage(content=content, usage_metadata=usage_metadata)
        return message

    async def ainvoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input, *args, **kwargs))


class OpenRouterModel(BaseArchytasModel):
    """Archytas backend model for OpenRouter using direct REST calls."""
    DEFAULT_MODEL = "openrouter/auto"
    api_key: str = ""

    def auth(self, **kwargs) -> None:
        self.api_key = (
            kwargs.get("api_key")
            or getattr(self.config, "api_key", None)
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        if not self.api_key:
            raise AuthenticationError("No OpenRouter API Key found. Set OPENROUTER_API_KEY or pass api_key.")

    def initialize_model(self, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        model_name = getattr(self.config, "model_name", None) or self.DEFAULT_MODEL
        return ChatOpenRouter(model=str(model_name), api_key=self.api_key)

    async def get_num_tokens_from_messages(
        self,
        messages: "list[BaseMessage]",
        tools: Optional[Sequence] = None,
    ) -> int:
        try:
            return self._model.get_num_tokens_from_messages(messages=messages, tools=tools)
        except Exception:
            return 0

    # Tool/function-calling is not yet implemented for OpenRouter in this wrapper.
    # bind_tools is accepted by the adapter but ignored.

    @lru_cache()
    def contextsize(self, model_name: Optional[str] = None) -> int | None:
        name = model_name or self.model_name
        default_value = 200_000
        if model_name is not None:
            try:
                return attributes_map[cast(ModelName, model_name)].context_size
            except KeyError:
                pass
        # Fallback default for safety so summarization threshold is usable
        logging.warning(f"OpenRouter context size unknown for model '{name}' (this implies model is not officially listed in archytas/models/openrouter_models.py, i.e. consider regenerating the file with `create_models_types_file()`). Using default context size: {default_value}.")
        return default_value