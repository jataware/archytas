import pdb
import os
import asyncio
import requests
import json
from typing import Generator, Literal, overload, TypedDict, Any, Optional, Sequence, cast
from typing_extensions import NotRequired
from pathlib import Path
from functools import lru_cache
import logging

here = Path(__file__).parent


from .openrouter_models import ModelName, attributes_map
from .base import BaseArchytasModel, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolCall as LangChainToolCall, ToolMessage, FunctionMessage
Role = Literal["user", "assistant", 'system', 'tool']

class Message(TypedDict):
    role: Role
    content: str
    tool_calls: 'NotRequired[list[OpenRouterToolCall]]'
    tool_call_id: NotRequired[str]

class OpenRouterToolResponse(TypedDict):
    thought: str
    tool_calls: 'list[OpenRouterToolCall]'

class OpenRouterToolCall(TypedDict):
    id: str
    type: Literal["function"]  # TODO: other types?
    function: 'OpenRouterToolFunction'

class OpenRouterToolFunction(TypedDict):
    name: str
    arguments: str # needs to be converted to dict via json.loads

def to_langchain_tool_call(tool_call: OpenRouterToolCall) -> LangChainToolCall:
    return LangChainToolCall(name=tool_call['function']['name'], args=json.loads(tool_call['function']['arguments']), id=tool_call['id'], type="tool_call")

def to_openrouter_tool_call(tool_call: LangChainToolCall) -> OpenRouterToolCall:
    return OpenRouterToolCall(id=tool_call['id'] or '', type="function", function=OpenRouterToolFunction(name=tool_call["name"], arguments=json.dumps(tool_call["args"])))

def pretty_tool_call(tool_call: OpenRouterToolCall) -> str:
    """Return a string representation of the tool call, i.e. `tool_name(arg1=value1, arg2=value2, ...)`"""
    args = json.loads(tool_call["function"]["arguments"])
    args_str = ', '.join([f'{k}={v}' for k, v in args.items()])
    return f'{tool_call["function"]["name"]}({args_str})'

class Model:
    """A simple class for talking to OpenRouter models directly via requests to the API"""
    def __init__(self, model:ModelName, openrouter_api_key:str, allow_parallel_tool_calls:bool=False):
        self.model = model
        self.openrouter_api_key = openrouter_api_key
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

        # updated after every completion
        self._usage_metadata: dict|None = None


    @overload
    def complete(self, messages: list[Message], *, stream:Literal[False]=False, tools:list|None=None, **kwargs) -> str | OpenRouterToolResponse: ...
    @overload
    def complete(self, messages: list[Message], *, stream:Literal[True], tools:list|None=None, **kwargs) -> Generator[str | OpenRouterToolResponse, None, None]: ...
    def complete(self, messages: list[Message], *, stream:bool=False, tools:list|None=None, **kwargs) -> str | OpenRouterToolResponse | Generator[str | OpenRouterToolResponse, None, None]:
        if stream:
            return self._streaming_complete(messages, tools, **kwargs)
        else:
            return self._blocking_complete(messages, tools, **kwargs)


    def _blocking_complete(self, messages: list[Message], tools:list|None=None, **kwargs) -> str | OpenRouterToolResponse:
        tool_payload = {"tools": tools, "parallel_tool_calls": self.allow_parallel_tool_calls} if tools else {}
        payload = {"model": self.model, "messages": messages, **tool_payload, **kwargs}
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            data=json.dumps(payload)
        )
        data = response.json()
        try:
            # TODO: handle case where server returns an error as a response
            self._usage_metadata = data['usage']
            if 'tool_calls' in data['choices'][0]['message'] and len(data['choices'][0]['message']['tool_calls']) > 0:
                return OpenRouterToolResponse(thought=data['choices'][0]['message']['content'], tool_calls=data['choices'][0]['message']['tool_calls'])
            return data['choices'][0]['message']['content']
        except KeyError as e:
            raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
        except Exception as e:
            raise ValueError(f"An error occurred while processing the response: '{data}'. {e}") from e

    # TODO: should request timeout be a setting rather than hardcoded?
    # TODO: streaming with tools not handled...
    def _streaming_complete(self, messages: list[Message], tools:list|None=None, **kwargs) -> Generator[str|OpenRouterToolResponse, None, None]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        tool_payload = {"tools": tools, "parallel_tool_calls": False} if tools else {}
        payload = {"model": self.model, "messages": messages, "stream": True, **tool_payload, **kwargs}

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            buf = []
            for line in r.iter_lines(decode_unicode=True, chunk_size=1024):
                line = cast(str|None, line)
                if line is None:
                    continue
                
                if line.startswith("data:"):
                    buf.append(line[5:].lstrip())
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
                        obj = json.loads(data)  #TODO: make a typed dict to annotate the response here
                    except json.JSONDecodeError:
                        continue # wait for the next complete event
                    try:
                        content: str|None = obj["choices"][0]["delta"].get("content")
                        tool_calls: list[OpenRouterToolCall]|None = obj["choices"][0]["delta"].get("tool_calls")
                        if tool_calls:
                            yield OpenRouterToolResponse(thought=content or '', tool_calls=tool_calls)
                        elif content:
                            yield content
                    except KeyError as e:
                        raise ValueError(f"Unexpected response format: '{data}'. Please check the API response. {e}") from e
                    
                    # update the usage metadata (typically on the final chunk)
                    if "usage" in obj:
                        self._usage_metadata = obj["usage"]
                    
                    continue

                
                # ignore other fields like "event:" / "id:" / comments / etc.




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




# from .model import Model, Message, Role
# from typing import Literal, Generator, overload

class Agent:
    """Basically just a model paired with message history tracking"""
    def __init__(self, model: Model, tools:list|None=None):
        self.model = model
        self.messages: list[Message] = []
        self.tools = tools

    @overload
    def add_message(self, *, role: Role, content: str): ...
    @overload
    def add_message(self, *, role: Role, content: str, tool_call_id: str): ...
    @overload
    def add_message(self, *, role: Role, content: str, tool_calls: list[OpenRouterToolCall]): ...
    def add_message(self, *, role: Role, content: str, tool_calls: list[OpenRouterToolCall]|None=None, tool_call_id: str|None=None):
        assert tool_calls is None or tool_call_id is None, "tool_calls and tool_call_id cannot both be provided"
        if tool_calls:
            message = Message(role=role, content=content, tool_calls=tool_calls)
        elif tool_call_id:
            message = Message(role=role, content=content, tool_call_id=tool_call_id)
        else:
            message = Message(role=role, content=content)
        self.messages.append(message)
    
    def add_user_message(self, content: str):
        self.add_message(role='user', content=content)

    def add_assistant_message(self, content: str):
        self.add_message(role='assistant', content=content)
    
    def add_assistant_tool_calls(self, content: str, tool_calls: list[OpenRouterToolCall]):
        self.add_message(role='assistant', content=content, tool_calls=tool_calls)

    def add_tool_message(self, tool_call_id: str, content: str):
        self.add_message(role='tool', tool_call_id=tool_call_id, content=content)

    def add_system_message(self, content: str):
        self.add_message(role='system', content=content)
    
    @overload
    def execute(self, stream:Literal[False]=False) -> str|OpenRouterToolResponse: ...
    @overload
    def execute(self, stream:Literal[True]) -> Generator[str | OpenRouterToolResponse, None, None]: ...
    def execute(self, stream:bool=False) -> str | OpenRouterToolResponse | Generator[str | OpenRouterToolResponse, None, None]:
        if stream:
            return self._streaming_execute()
        else:
            return self._blocking_execute()
    
    def _blocking_execute(self) -> str | OpenRouterToolResponse:
        result = self.model.complete(self.messages, stream=False, tools=self.tools)
        if isinstance(result, str):
            self.add_assistant_message(result)
        else:
            self.add_assistant_tool_calls(result['thought'], result['tool_calls'])
        return result

    def _streaming_execute(self) -> Generator[str|OpenRouterToolResponse, None, None]:
        # stream the chunks while also capturing them
        result_chunks = []
        tool_calls = []
        for chunk in self.model.complete(self.messages, stream=True, tools=self.tools):
            if isinstance(chunk, dict):
                tool_calls.append(chunk)
            else:
                result_chunks.append(chunk)
            yield chunk
        
        # add the message to the history after streaming is done
        if tool_calls:
            self.add_assistant_tool_calls(''.join(result_chunks), tool_calls)
        else:
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
        # Delegate to the minimal client
        try:
            self._client = Model(model=cast(ModelName, model), openrouter_api_key=api_key)
        except Exception:
            # If model isn't recognized in our Literal list, still construct with raw string
            self._client = Model(model=cast("ModelName", model), openrouter_api_key=api_key)  # type: ignore[arg-type]

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

    def _convert_messages(self, messages: list[BaseMessage]) -> list[Message]:
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

        converted: list = []#[Message] = []
        for msg in messages:
            match msg:
                case HumanMessage():
                    converted.append({"role": "user", "content": serialize_content(msg.content)})
                case AIMessage(tool_calls=list() as tool_calls):
                    converted.append({"role": "assistant", 'content': msg.content, 'tool_calls': list(map(to_openrouter_tool_call, tool_calls))})
                case AIMessage():
                    pdb.set_trace()
                    ...
                case SystemMessage():
                    converted.append({"role": "system", "content": serialize_content(msg.content)})
                case ToolMessage():
                    # convert tool call result back to openrouter dict response format
                    converted.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
                case FunctionMessage():
                    pdb.set_trace()
                    converted.append({"role": "system", "content": f"Observation from {getattr(msg, 'name', 'function')}: {serialize_content(msg.content)}"})
                case _:
                    pdb.set_trace()
                    converted.append({"role": "user", "content": serialize_content(msg.content)})
        return converted

    def get_num_tokens_from_messages(self, *, messages: list[BaseMessage], tools: Optional[Sequence[Any]] = None) -> int:
        # TODO: pull the actual token count from the OpenRouter API... this is just a placeholder
        # Rough heuristic: 4 chars per token
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            else:
                total_chars += len(json.dumps(msg.content))
        return max(1, total_chars // 4)

    def invoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        # Convert LangChain messages to OpenRouter format
        converted_messages = self._convert_messages(input)

        # # Breakpoint if generation params are requested; minimal Model doesn't support them yet
        # unsupported_gen_params = {k: v for k, v in kwargs.items() if k in ("temperature", "top_p", "max_tokens", "stop", "response_format", "seed") and v is not None}
        # if unsupported_gen_params:
        #     pdb.set_trace()  # Missing: forward generation params to OpenRouter. Update Model.complete to accept and include these in the request payload.

        # Breakpoint if tools are bound; minimal Model doesn't handle tool schemas/calls
        # if getattr(self, "_schemas", None):
        #     pdb.set_trace()  # Missing: pass tool schemas and enable tool/function calling. Extend Model.complete to accept `tools` and `tool_choice`.

        response = self._client.complete(converted_messages, stream=False, tools=self._schemas, **kwargs)

        assert self._client._usage_metadata is not None, "INTERNAL ERROR: Usage metadata was not set for previous completion call"
        usage_metadata = {
            'input_tokens': self._client._usage_metadata['prompt_tokens'],
            'output_tokens': self._client._usage_metadata['completion_tokens'],
            'total_tokens': self._client._usage_metadata['total_tokens']
        }

        if isinstance(response, str):
            return AIMessage(content=response, usage_metadata=usage_metadata)
        elif isinstance(response, dict):
            return AIMessage(content=response['thought'], tool_calls=list(map(to_langchain_tool_call, response['tool_calls'])), usage_metadata=usage_metadata)
            pdb.set_trace()
            # TODO: find the correct tool message class to return
            #       probably need to reshape the response to match the schema 
            return TODO_ToolMessage(content=response['thought'], usage_metadata=usage_metadata, tool_calls=response['tool_calls'])
        else:
            pdb.set_trace()
            raise ValueError(f"Unexpected response type: {type(response)}")

        # match response:
        #     case str():
        #         return AIMessage(content=response, usage_metadata=usage_metadata)
        #     case dict():
        #         return AIMessage(content=response['thought'], usage_metadata=usage_metadata)

        # return AIMessage(content=content_text, usage_metadata=usage_metadata)

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

    
    # def _preprocess_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
    #     pdb.set_trace()
    #     ...
    #     raise NotImplementedError("Preprocessing messages is not implemented for OpenRouter")

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