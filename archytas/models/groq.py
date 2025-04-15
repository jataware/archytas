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

import logging
logger = logging.getLogger(__name__)

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

CRITICAL: Do NOT embed tool calls inside the `final_answer` response. The `final_answer` should be the FINAL response to the user, not another tool call.
If you need to run more code or use other tools, do that BEFORE calling `final_answer`.

CRITICAL: Unless explicitly instructed to do so by the user, you MUST NEVER put your code into your content or messages. You must use the `run_code` tool to run code.
You MUST format this properly as a tool call.

CRITICAL: if you have access to the `run_code` tool, you MUST use it to run code. If you are asked to do a coding task, do NOT simply
provide by the code and ask whether or not to proceed: you have been given the task, now DO IT! USE `run_code` to do so! This is of the utmost importance!
If you run into an traceback or other code execution error with the `run_code` tool, think about the problem then you MUST debug it with another invocation of the `run_code` tool.

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
        # Process ToolMessages in the input
        for i, msg in enumerate(input):
            if isinstance(msg, ToolMessage):
                input[i] = self.handle_groq(msg)
                logger.debug(f"Processed ToolMessage in input at position {i}")
        
        # Call the parent's ainvoke method
        response = await super().ainvoke(input, config=config, stop=stop, **kwargs)
        
        # Process the response
        logger.debug(f"Processing response from ainvoke: {type(response).__name__}")
        processed_response = self.handle_groq(response)
        
        # Log the processed response
        if hasattr(processed_response, "tool_calls") and processed_response.tool_calls:
            logger.debug(f"Processed response has {len(processed_response.tool_calls)} tool_calls")
        else:
            logger.debug("Processed response does not have tool_calls attribute or it's empty")
        
        return processed_response

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

    def _rectify_result(self, response_message: "AIMessage"):
        """
        Ensure that tool calls in the content are properly extracted and set on the message object.
        This is called by the ReAct agent before storing the message in the chat history.
        """
        # First call the parent's _rectify_result method
        response_message = super()._rectify_result(response_message)
        
        # Process the response using handle_groq
        result = self.handle_groq(response_message)
        return result
    
    @lru_cache()
    def contextsize(self, model_name: Optional[str]=None) -> int | None:
        if model_name is None:
            model_name = self.model_name
        model_list = self._client.models.list()
        model_index = {model.id: model for model in model_list.data}
        model_info = model_index.get(model_name, None)
        context_window = getattr(model_info, "context_window", None)
        return context_window
    
    def handle_groq(self, response):
        """
        Process a response from Groq to ensure tool calls are properly formatted.
        This is called from both ainvoke (for responses) and _rectify_result (for message history).
        """
        # Log the response details
        logger.debug(f"Handle Groq called with response type: {type(response).__name__}")
        
        # Check if this response has already been processed
        if hasattr(response, "_groq_processed") and response._groq_processed:
            logger.debug("Response has already been processed, skipping")
            return response
        
        # Log the response ID or other identifier
        if hasattr(response, "id"):
            logger.debug(f"Processing response with ID: {response.id}")
        
        # Check if response already has tool_calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.debug(f"Response already has {len(response.tool_calls)} tool_calls")
        else:
            logger.debug("Response does not have tool_calls attribute or it's empty")
            # Initialize tool_calls if it doesn't exist
            if not hasattr(response, "tool_calls"):
                response.tool_calls = []
        
        # Handle ToolMessage differently
        if isinstance(response, ToolMessage):
            self._process_tool_message(response)
            return response
        
        # For AIMessage, extract tool calls from various sources
        extracted_tool_calls = []
        
        # 1. Extract from additional_kwargs
        if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            extracted_tool_calls.extend(self._extract_from_additional_kwargs(response))
        
        # 2. Extract from content
        if hasattr(response, 'content') and response.content and isinstance(response.content, str) and '<tool_call>' in response.content:
            content_tool_calls = self._extract_from_content(response)
            if content_tool_calls:
                extracted_tool_calls.extend(content_tool_calls)
        
        # 3. Process embedded tool calls in final_answer
        if response.tool_calls:
            response.tool_calls = self._process_embedded_tool_calls(response.tool_calls)
        
        # 4. Add extracted tool calls to response
        if extracted_tool_calls:
            response.tool_calls = extracted_tool_calls
            response.tool_call_id = extracted_tool_calls[0]["id"]
            logger.debug(f"Set tool_call_id={response.tool_call_id} on response")
        # 5. Create a tool call if there's a tool_call_id but no tool_calls
        elif hasattr(response, "tool_call_id") and response.tool_call_id and not response.tool_calls:
            self._create_tool_call_from_id(response)
        
        # Final check to ensure tool_calls is set
        if not response.tool_calls:
            logger.debug("Response still does not have tool_calls after processing")
        else:
            logger.debug(f"Response now has {len(response.tool_calls)} tool_calls after processing")
        
        # Mark as processed to prevent double-processing
        response._groq_processed = True
        
        return response

    def _process_tool_message(self, response):
        """Process a ToolMessage to ensure it has tool_calls."""
        logger.debug(f"Processing ToolMessage with tool_call_id={response.tool_call_id}")
        
        # Extract tool name from artifact if available
        tool_name = "unknown"
        if hasattr(response, "artifact") and isinstance(response.artifact, dict) and "tool_name" in response.artifact:
            tool_name = response.artifact["tool_name"]
            logger.debug(f"Extracted tool_name={tool_name} from artifact")
        
        # Add a tool call with the same ID if not already present
        if not any(tc.get("id") == response.tool_call_id for tc in response.tool_calls):
            response.tool_calls.append({
                "id": response.tool_call_id,
                "name": tool_name,
                "args": {},
                "type": "tool_call"
            })
            logger.debug(f"Added tool_call with ID={response.tool_call_id} to ToolMessage")
        
        # Mark as processed
        response._groq_processed = True
        return response

    def _extract_from_additional_kwargs(self, response):
        """Extract tool calls from additional_kwargs."""
        tool_calls = []
        for tc in response.additional_kwargs["tool_calls"]:
            tc_id = tc.get("id")
            if tc_id:
                function_data = tc.get("function", {})
                name = function_data.get("name")
                arguments = function_data.get("arguments", "{}")
                
                if isinstance(arguments, str):
                    try:
                        args = json.loads(arguments)
                    except json.JSONDecodeError:
                        args = {"raw_arguments": arguments}
                else:
                    args = arguments
                
                formatted_tc = {
                    "id": tc_id,
                    "name": name,
                    "args": args,
                    "type": "tool_call"
                }
                tool_calls.append(formatted_tc)
                logger.debug(f"Extracted tool call from additional_kwargs: {tc_id}")
        return tool_calls

    def _extract_from_content(self, response):
        """Extract tool calls from content field."""
        tool_calls = []
        content = response.content
        
        try:
            # Extract the JSON content
            tool_call_parts = content.split('<tool_call>')
            if len(tool_call_parts) > 1:
                tool_call_json = tool_call_parts[1]
                
                # Handle potential end markers
                end_markers = ['<｜tool▁calls▁end｜>', '</tool_call>']
                for marker in end_markers:
                    if marker in tool_call_json:
                        tool_call_json = tool_call_json.split(marker)[0]
                
                # Clean up any trailing text that might not be part of the JSON
                last_brace_index = tool_call_json.rfind('}')
                if last_brace_index > 0:
                    tool_call_json = tool_call_json[:last_brace_index+1]
                
                # Parse the JSON
                tool_call_data = json.loads(tool_call_json)
                
                # Create a properly formatted tool call
                import uuid
                tc_id = tool_call_data.get("id")
                if not tc_id:
                    tc_id = f"call_{uuid.uuid4().hex[:4]}"
                
                name = tool_call_data.get("name")
                arguments = tool_call_data.get("arguments", {})
                
                formatted_tc = {
                    "id": tc_id,
                    "name": name,
                    "args": arguments,
                    "type": "tool_call"
                }
                
                if isinstance(formatted_tc["args"], str):
                    try:
                        formatted_tc["args"] = json.loads(formatted_tc["args"])
                    except json.JSONDecodeError:
                        formatted_tc["args"] = {"raw_arguments": formatted_tc["args"]}
                
                tool_calls.append(formatted_tc)
                logger.warning(f"Extracted tool call from content: {tc_id}, name: {name}")
                
                # Clear the content to prevent double-processing
                response.content = tool_call_parts[0]
                if len(tool_call_parts) > 2:
                    for part in tool_call_parts[2:]:
                        for marker in end_markers:
                            if marker in part:
                                part = part.split(marker, 1)[1]
                        response.content += part
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in tool call: {str(e)}")
            # Try a more aggressive approach to extract the JSON
            try:
                import re
                # Look for tool call patterns with balanced braces
                # This is a more complex regex that handles nested JSON objects
                pattern = r'<tool_call>(.*?)(?:<｜tool▁calls▁end｜>|</tool_call>)'
                matches = re.findall(pattern, content, re.DOTALL)
                
                if matches:
                    # Process each match
                    cleaned_content = content
                    for match in matches:
                        try:
                            # Find the balanced JSON object
                            json_str = match.strip()
                            # Ensure we have a complete JSON object
                            open_braces = 0
                            close_braces = 0
                            for char in json_str:
                                if char == '{':
                                    open_braces += 1
                                elif char == '}':
                                    close_braces += 1
                            
                            # If we have unbalanced braces, try to fix it
                            if open_braces > close_braces:
                                json_str += '}' * (open_braces - close_braces)
                            
                            # Parse the JSON
                            tool_call_data = json.loads(json_str)
                            
                            # Create a properly formatted tool call
                            import uuid
                            tc_id = tool_call_data.get("id")
                            if not tc_id:
                                tc_id = f"call_{uuid.uuid4().hex[:4]}"
                            
                            name = tool_call_data.get("name")
                            arguments = tool_call_data.get("arguments", {})
                            
                            formatted_tc = {
                                "id": tc_id,
                                "name": name,
                                "args": arguments,
                                "type": "tool_call"
                            }
                            
                            if isinstance(formatted_tc["args"], str):
                                try:
                                    formatted_tc["args"] = json.loads(formatted_tc["args"])
                                except json.JSONDecodeError:
                                    formatted_tc["args"] = {"raw_arguments": formatted_tc["args"]}
                            
                            tool_calls.append(formatted_tc)
                            logger.warning(f"Extracted tool call using regex: {tc_id}, name: {name}")
                            
                            # Remove this tool call from the content
                            full_match = f'<tool_call>{match}'
                            for marker in ['<｜tool▁calls▁end｜>', '</tool_call>']:
                                if marker in content and content.find(full_match) < content.find(marker):
                                    full_match += marker
                                    break
                            
                            cleaned_content = cleaned_content.replace(full_match, '')
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON in match: {e}")
                            continue
                    
                    # Update the content
                    response.content = cleaned_content
                    logger.warning(f"Cleaned content after extracting tool calls")
                else:
                    logger.warning("No tool call matches found with regex")
            except Exception as e:
                logger.error(f"Failed to extract tool call using regex: {str(e)}")
        
        return tool_calls

    def _process_embedded_tool_calls(self, tool_calls):
        """Process tool calls embedded in final_answer."""
        new_tool_calls = []
        for tc in tool_calls:
            if tc.get("name") == "final_answer" and tc.get("args") and isinstance(tc.get("args"), dict):
                response_text = tc.get("args").get("response", "")
                if "<tool_call>" in response_text:
                    logger.debug("Detected tool call embedded in final_answer! Converting to proper tool call.")
                    
                    # Extract the embedded tool call
                    parts = response_text.split("<tool_call>")
                    prefix_text = parts[0].strip()
                    
                    try:
                        # Extract the JSON content
                        tool_call_json = parts[1]
                        if "<｜tool▁calls▁end｜>" in tool_call_json:
                            tool_call_json = tool_call_json.split("<｜tool▁calls▁end｜>")[0]
                        
                        # Parse the JSON
                        tool_call_data = json.loads(tool_call_json)
                        
                        # Create a properly formatted tool call
                        import uuid
                        tc_id = f"call_{uuid.uuid4().hex[:4]}"
                        
                        extracted_tc = {
                            "id": tc_id,
                            "name": tool_call_data.get("name"),
                            "args": tool_call_data.get("arguments", {}),
                            "type": "tool_call"
                        }
                        
                        if isinstance(extracted_tc["args"], str):
                            try:
                                extracted_tc["args"] = json.loads(extracted_tc["args"])
                            except json.JSONDecodeError:
                                extracted_tc["args"] = {"raw_arguments": extracted_tc["args"]}
                        
                        # Add the extracted tool call to our list
                        new_tool_calls.append(extracted_tc)
                        logger.debug(f"Extracted embedded tool call: {extracted_tc['name']}")
                        
                        # If there was text before the tool call, keep the final_answer with just that text
                        if prefix_text:
                            tc["args"]["response"] = prefix_text
                            new_tool_calls.append(tc)
                            logger.debug(f"Kept final_answer with prefix text: {prefix_text[:50]}...")
                        # Otherwise, we don't need the final_answer at all
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.error(f"Error extracting embedded tool call: {e}")
                        # Keep the original tool call if we couldn't extract the embedded one
                        new_tool_calls.append(tc)
                else:
                    # No embedded tool call, keep the original
                    new_tool_calls.append(tc)
            else:
                # Not a final_answer or no embedded tool call, keep the original
                new_tool_calls.append(tc)
        
        return new_tool_calls

    def _create_tool_call_from_id(self, response):
        """Create a tool call from tool_call_id."""
        # Extract tool name from artifact if available
        tool_name = "unknown"
        if hasattr(response, "artifact") and isinstance(response.artifact, dict) and "tool_name" in response.artifact:
            tool_name = response.artifact["tool_name"]
        
        # Add a tool call with the same ID
        response.tool_calls.append({
            "id": response.tool_call_id,
            "name": tool_name,
            "args": {},
            "type": "tool_call"
        })
        logger.debug(f"Created tool_call with ID={response.tool_call_id} to match tool_call_id")