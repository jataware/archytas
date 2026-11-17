import json
import logging
import os
import re
import uuid
from typing import ClassVar

from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrockConverse

from ...exceptions import AuthenticationError
from ..families import ModelFamily, AnthropicModelFamily, LlamaModelFamily, OpenAIModelFamily
from .base import ModelProvider

DEFERRED_TOKEN_VALUE = "***deferred***"

STOP_REASON_MESSAGES = {
    "max_tokens": "The model reached its output token limit before completing its response.",
    "guardrail_intervened": "The response was stopped by a Bedrock guardrail.",
    "content_filtered": "The response was stopped by Bedrock content filtering.",
    "malformed_model_output": "The model produced an invalid response and could not complete the request.",
    "malformed_tool_use": "The model produced an invalid tool call and could not complete the request.",
    "model_context_window_exceeded": "The conversation exceeded the model's context window.",
}




class BedrockProvider(ModelProvider):
    MODEL_SPECIFIC_CONTENT_TYPES = frozenset({"reasoning_content"})
    DEFAULT_REGION = "us-east-1"

    BEDROCK_MODEL_FAMILY_REGEX_MAP: ClassVar[dict[str, type[ModelFamily]]] = {
        "anthropic": AnthropicModelFamily,
        "openai": OpenAIModelFamily,
        "meta.llama3": LlamaModelFamily,
        "meta.llama4": LlamaModelFamily,
    }

    def parse_prefix(self, model_name):
        family, name = super().parse_prefix(model_name)
        if family is not None:
            return family, name
        # Foundation model and inference profile ARNs can include a recognizable ID.
        identifier = name.rsplit("/", 1)[-1]
        family_regex = "|".join(re.escape(key) for key in self.BEDROCK_MODEL_FAMILY_REGEX_MAP.keys())
        regex_search = r"^(?:(?:us|eu|ap|global)\.)?(" + family_regex + r")[\.-]"
        match = re.match(regex_search, identifier, re.IGNORECASE)
        if match:
            family = self.BEDROCK_MODEL_FAMILY_REGEX_MAP[match.group(1).lower()]
        return family, name

    def auth(self, model, **kwargs) -> None:
        for attribute in (
            'credentials_profile_name', 'aws_access_key', 'aws_secret_key', 'aws_session_token',
        ):
            if not hasattr(model, attribute):
                setattr(model, attribute, None)
        if model.config.model_extra:
            access = model.config.model_extra.get('aws_access_key', '')
            secret = model.config.model_extra.get('aws_secret_key', '')
            session = model.config.model_extra.get('aws_session_token', '')
            if access != '' and secret != '':
                model.aws_access_key = access
                model.aws_secret_key = secret
                model.aws_session_token = session if session != '' else None

        # not handled - running on EC2 and expecting to authenticate via instance profile and IMDSv2
        # TODO: handle ec2/instance profile if we need it later. could be as easy as removing the exception below
        if 'credentials_profile_name' in kwargs:
            model.credentials_profile_name = kwargs['credentials_profile_name']
            return

        # required if not using credentials or env vars. manually passing the argument should take
        # precedence over env vars
        aws_keys = ['aws_access_key', 'aws_secret_key']
        if any([(key in kwargs) for key in aws_keys + ['aws_session_token']]):
            for key in aws_keys:
                if key in kwargs:
                    setattr(model, key, kwargs[key])
                else:
                    raise AuthenticationError(f'one of aws_access_key or aws_secret_key was missing: Missing key: {key}')
            # NOT required, but if present, the above two also *must* exist.
            model.aws_session_token = kwargs.get('aws_session_token', None)
            return

        # A complete configured pair must not be replaced by environment keys.
        if model.aws_access_key and model.aws_secret_key:
            return

        env_vars = {
            'AWS_ACCESS_KEY_ID': 'aws_access_key',
            'AWS_SECRET_ACCESS_KEY': 'aws_secret_key'
        }
        if any([(var in os.environ) for var in list(env_vars.keys()) + ['AWS_SESSION_TOKEN']]):
            for var in env_vars:
                if var in os.environ:
                    setattr(model, env_vars[var], os.environ[var])
                else:
                    raise AuthenticationError(f'missing one of required env vars: access_key or secret_key: {var}')
            model.aws_session_token = os.environ.get('AWS_SESSION_TOKEN', None)

    def initialize_model(self, model, **kwargs):
        region = os.environ.get('AWS_REGION', getattr(model, 'DEFAULT_REGION', self.DEFAULT_REGION))
        max_tokens = None
        if model.config.model_extra:
            if model.config.model_extra.get('region', '') != '':
                region = model.config.model_extra.get('region')
            max_tokens = model.config.model_extra.get('max_tokens', None)
        name = model.config.model_name or model.DEFAULT_MODEL
        if model.credentials_profile_name:
            return ChatBedrockConverse(
                credentials_profile_name=model.credentials_profile_name,
                region_name=region,
                model=name,
                max_tokens=max_tokens
            )
        return ChatBedrockConverse(
            aws_access_key_id=model.aws_access_key,
            aws_secret_access_key=model.aws_secret_key,
            aws_session_token=model.aws_session_token or None,
            region_name=region,
            model=name,
            max_tokens=max_tokens
        )

    def preprocess_messages(self, model, messages):
        # Only Claude uses these markers. Its message ordering is supplied by
        # the family; the marker encoding is specific to the provider API.
        if not isinstance(model.model_family, AnthropicModelFamily):
            return messages
        from .anthropic import _attach_cache_control, _prompt_cache_disabled
        output = list(messages)
        if not _prompt_cache_disabled():
            tail_count = getattr(model, "tail_injection_count", 0) or 0
            last_persisted_idx = len(output) - tail_count - 1
            if last_persisted_idx > 0:
                output[last_persisted_idx] = _attach_cache_control(output[last_persisted_idx])
        return output

    async def ainvoke(self, model, invoke, input, *, config=None, stop=None, **kwargs):
        response = await invoke(input, config=config, stop=stop, **kwargs)
        stop_reason = response.response_metadata.get("stopReason")
        interrupted = stop_reason in STOP_REASON_MESSAGES
        if interrupted or (stop_reason == "end_turn" and not response.tool_calls):
            if isinstance(response.content, str):
                text = response.content
            else:
                text = "\n".join(
                    block["text"] for block in response.content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if interrupted:
                # Partial or malformed tool calls cannot be replayed or executed.
                if isinstance(response.content, list):
                    response.content = [
                        block for block in response.content
                        if not isinstance(block, dict) or (
                            block.get("type") not in ("tool_use", "tool_call")
                            and "toolUse" not in block
                        )
                    ]
                response.tool_calls.clear()
                response.invalid_tool_calls.clear()
                if not text.strip():
                    text = STOP_REASON_MESSAGES[stop_reason]
                    response.content = text
            response.tool_calls.append({
                "id": uuid.uuid4().hex,
                "name": "final_answer",
                "args": {"response": text},
            })
        return response

    def handle_invoke_error(self, model, error: BaseException):
        # client error catches a lot of credentials errors like incorrect profile names
        if isinstance(error, NoCredentialsError) or (isinstance(error, ClientError) and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403):
            raise AuthenticationError(f"{error}")
        if model.last_sent_messages:
            try:
                message_output = [msg.model_dump() for msg in model.last_sent_messages]
                logging.warning(
                    "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
                    json.dumps(message_output, indent=2)
                )
            except Exception:
                # Diagnostic logging must never mask the original error.
                pass
        raise error

    def fallback_contextsize(self, model, model_name=None):
        return 50_000

