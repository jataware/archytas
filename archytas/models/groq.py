import json
import re
import groq.resources
import requests
from functools import lru_cache
from typing import Optional, cast, ClassVar

import groq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel as PydanticModel, Field

from .base import BaseArchytasModel, set_env_auth, ModelConfig

class Structure(PydanticModel):
    thought: str = Field(..., description="Thought process of why you are calling the tool")
    tool: str
    tool_input: str | dict | list
    helpful_thoguht: bool

class GroqModel(BaseArchytasModel):
    model: ChatGroq
    _client: groq.Groq
    api_key: str = ""

    DEFAULT_MODEL: ClassVar[str] = "llama3-8b-8192"
    MODEL_PROMPT_INSTRUCTIONS: str = """\
CRITICAL: You MUST ALWAYS use the `final_answer` tool to report back to the user at the end of a ReAct loop.
    
CRITICAL: You MUST properly format your tool calls. Do NOT forget this! It is absolutely critical, otherwise your tool calls will not be executed.
Do NOT try to add tool calls to the text of your responses via `<tool_call>` or similar text. You must properly add them as `tool_calls` in the response.

When generating JSON, remember to not wrap strings in triple quotes such as \'\'\' or \"\"\". If you want to add newlines \
to the JSON text, use `\\n` to add newlines.
Ensure all generated JSON is valid and would pass a JSON validator.
"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = cast(groq.resources.chat.Completions, self.model.client)._client

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        else:
            self.api_key = self.config.api_key
        if not self.api_key:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        model = ChatGroq(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.api_key,
            base_url="https://api.groq.com/"
        )
        return model

    @property
    def model_name(self) -> str | None:
        model_name = getattr(self.model, "model_name", None)
        if model_name is not None:
            return model_name
        else:
            return getattr(self.config, "model_name", self.DEFAULT_MODEL)

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        """
        Invoke the model asynchronously and reformat the response to match Anthropic's format.
        """
        # Check if this is a response to a tool execution that had an error
        tool_error = None
        execution_failed = False
        
        # Track if we've already forced an error acknowledgment
        forced_error_acknowledgment = False
        
        for message in input:
            if hasattr(message, 'content') and isinstance(message.content, str):
                # Check if we've already forced an error acknowledgment
                if "I encountered an error when executing the code:" in message.content and "Let me try a different approach." in message.content:
                    forced_error_acknowledgment = True
                
                # Check if execution explicitly failed
                if 'Successful?: False' in message.content:
                    execution_failed = True
                    # Look for error patterns in the message content
                    if 'ERROR:' in message.content or 'Traceback' in message.content or 'Exception:' in message.content:
                        tool_error = message.content
        
        # If we've already forced an error acknowledgment, modify the system prompt to encourage fixing the error
        if forced_error_acknowledgment:
            if config is None:
                config = {}
            
            # Add a reminder to fix the error
            for i, msg in enumerate(input):
                if isinstance(msg, SystemMessage):
                    if "You previously encountered an error. Please fix it by providing a new tool call." not in msg.content:
                        input[i].content += "\n\nYou previously encountered an error. Please fix it by providing a new tool call."
                    break
        
        response = await super().ainvoke(input, config=config, stop=stop, **kwargs)
        
        # If there was an error in the tool execution but the model is ignoring it,
        # we need to force the model to acknowledge the error
        if execution_failed and tool_error and hasattr(response, 'content') and not forced_error_acknowledgment:
            # Check if the model's response acknowledges the error
            error_acknowledged = False
            error_indicators = ['error', 'exception', 'failed', 'traceback', 'incorrect', 'issue', 'problem']
            
            # Check if any error indicators are in the response
            for indicator in error_indicators:
                if indicator.lower() in response.content.lower():
                    error_acknowledged = True
                    break
        
        # Check for tool calls embedded in text content
        if hasattr(response, 'content') and response.content:
            # Check if the content starts with <tool_call>
            if response.content.strip().startswith('<tool_call>'):
                try:
                    # Extract the JSON content between <tool_call> and the end
                    tool_call_json = response.content.strip()[len('<tool_call>'):]
                    
                    # Try to parse the JSON
                    tool_call_data = json.loads(tool_call_json)
                    
                    # Create a properly formatted tool call
                    if not hasattr(response, 'tool_calls') or not response.tool_calls:
                        response.tool_calls = []
                    
                    formatted_tool_call = {
                        "id": tool_call_data.get("id", f"call_{len(response.tool_calls)}"),
                        "name": tool_call_data.get("name"),
                        "args": tool_call_data.get("arguments", {}).copy(),
                        "type": "tool_call"
                    }
                    
                    # Ensure thought is present in args
                    if "thought" not in formatted_tool_call["args"] and "thought" in tool_call_data:
                        formatted_tool_call["args"]["thought"] = tool_call_data["thought"]
                    
                    response.tool_calls.append(formatted_tool_call)
                    
                    # Clear the content since we've extracted the tool call
                    response.content = ""
                    
                    import logging
                    logging.info(f"Extracted tool call from <tool_call> format")
                except json.JSONDecodeError as e:
                    import logging
                    logging.warning(f"Failed to parse tool call JSON: {e}")
                    logging.warning(f"JSON string: {tool_call_json}")
            
            # Extract all tool calls using a more robust pattern that handles incomplete tool calls
            tool_call_patterns = [
                r'<tool_call>(.*?)(?:<｜tool▁calls▁end｜>|</tool_call>)',  # Complete with end tag
                r'<tool_call>(.*?)$'  # Incomplete (runs to end of string)
            ]
            
            tool_call_matches = []
            for pattern in tool_call_patterns:
                matches = re.findall(pattern, response.content, re.DOTALL)
                if matches:
                    tool_call_matches.extend(matches)
                    break  # Use the first pattern that works
            
            if tool_call_matches:
                # Extract the text before the first tool call
                pre_tool_text = re.split(r'<tool_call>', response.content)[0].strip()
                
                # Parse the tool call JSON
                try:
                    if not hasattr(response, 'tool_calls') or response.tool_calls is None:
                        response.tool_calls = []
                    
                    for tool_call_json in tool_call_matches:
                        # Clean up the JSON string
                        tool_call_json = tool_call_json.strip()
                        if tool_call_json.endswith('<｜tool▁calls▁end｜>'):
                            tool_call_json = tool_call_json[:-len('<｜tool▁calls▁end｜>')]
                        if tool_call_json.endswith('</tool_call>'):
                            tool_call_json = tool_call_json[:-len('</tool_call>')]
                        
                        # Try to find a valid JSON object in the string
                        # This handles cases where the JSON is incomplete
                        json_match = re.search(r'(\{.*\})', tool_call_json)
                        if json_match:
                            tool_call_json = json_match.group(1)
                        
                        # Parse the JSON
                        tool_call_data = json.loads(tool_call_json)
                        
                        # Format the tool call properly
                        formatted_tool_call = {
                            "id": tool_call_data.get("id", f"call_{len(response.tool_calls)}"),
                            "name": tool_call_data.get("name"),
                            "args": tool_call_data.get("arguments", {}),
                            "type": "tool_call"
                        }
                        
                        # Ensure thought is present in args
                        if "thought" not in formatted_tool_call["args"] and "thought" in tool_call_data:
                            formatted_tool_call["args"]["thought"] = tool_call_data["thought"]
                        elif "thought" not in formatted_tool_call["args"]:
                            formatted_tool_call["args"]["thought"] = pre_tool_text
                        
                        response.tool_calls.append(formatted_tool_call)
                    
                    # Set content to the text before the tool call
                    response.content = pre_tool_text
                    
                    # Log that we extracted tool calls
                    import logging
                    logging.info(f"Extracted {len(tool_call_matches)} tool calls from text")
                except json.JSONDecodeError as e:
                    # If we can't parse the JSON, try to extract a valid JSON object
                    import logging
                    logging.warning(f"Failed to parse tool call JSON: {e}")
                    logging.warning(f"JSON string: {tool_call_json}")
        
        # For responses with tool calls, ensure they're properly formatted
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Add a text entry if there's any thought or explanation
            thought = ""
            for tool_call in response.tool_calls:
                if "thought" in tool_call.get("args", {}):
                    thought = tool_call["args"]["thought"]
                    break
            
            if not thought:
                # Create a generic thought based on the tool
                tool_name = response.tool_calls[0]["name"]
                if tool_name == "final_answer":
                    response_text = response.tool_calls[0]["args"].get("response", "")
                    thought = f"I'll provide a final answer: {response_text[:50]}..."
                    
                    # For final_answer tool calls, convert to direct text response
                    # This matches Anthropic's format and avoids the summarizer issue
                    response.content = response_text
                    response.tool_calls = []
                    
                    # Add stop_reason to match Anthropic's format
                    if 'response_metadata' not in response.__dict__:
                        response.response_metadata = {}
                    response.response_metadata['stop_reason'] = "end_turn"
                    
                    # Return early since we've converted to a text response
                    return response
                else:
                    thought = f"I'll use the {tool_name} tool to help with this task."
            
            # Set the response content to the thought text if it's empty
            if not response.content or response.content == "":
                response.content = thought
            
            # Ensure tool_calls are properly formatted for get_tool_caller
            formatted_tool_calls = []
            for tool_call in response.tool_calls:
                # Create a properly formatted tool call
                formatted_tool_call = {
                    "id": tool_call.get("id", f"call_{len(formatted_tool_calls)}"),
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args", {}).copy(),
                    "type": "tool_call"
                }
                
                # Ensure thought is present in args
                if "thought" not in formatted_tool_call["args"]:
                    formatted_tool_call["args"]["thought"] = thought
                
                formatted_tool_calls.append(formatted_tool_call)
            
            # Replace the tool_calls with the formatted version
            response.tool_calls = formatted_tool_calls
        
        return response

    def _preprocess_messages(self, messages: list[BaseMessage]):
        from ..agent import AutoContextMessage, ContextMessage
        output = []
        system_messages = []
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case AIMessage():
                    # Duplicate mesage so we don't change raw storage
                    msg = message.model_copy()
                    if msg.tool_calls:
                        msg.tool_calls.clear()
                    output.append(msg)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    @lru_cache()
    def contextsize(self, model_name: Optional[str]=None) -> int | None:
        if model_name is None:
            model_name = self.model_name
        model_list = self._client.models.list()
        model_index = {model.id: model for model in model_list.data}
        model_info = model_index.get(model_name, None)
        context_window = getattr(model_info, "context_window", None)
        return context_window