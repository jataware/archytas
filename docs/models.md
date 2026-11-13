# Model providers and families

An Archytas model combines two components:

- A `ModelProvider` handles authentication, client construction, API parameter
  encoding, transport-specific content, and SDK errors.
- A `ModelFamily` handles model instructions, message ordering, capability
  defaults, and model-specific interpretation of results.

The existing `OpenAIModel`, `AnthropicModel`, `BedrockModel`, `AzureOpenAIModel`,
`OpenRouterModel`, `GroqModel`, and `OllamaModel` classes remain available at their
existing import paths. They select default components and retain the facade used
by agents, chat history, tools, and token estimation. Existing subclass hooks,
including `auth`, `initialize_model`, `_preprocess_messages`, `_rectify_result`,
`process_result`, and `handle_invoke_error`, remain overrideable.

## Package layout

Implementations are grouped by responsibility:

```text
archytas/models/
    base.py                 # Composed model facade and ModelConfig
    providers/
        base.py             # ModelProvider interface
        openai.py           # Service-specific provider implementations
        bedrock.py
        ...
    families/
        base.py             # ModelFamily, reasoning settings, registry, helpers
        openai.py           # Family behavior implementations
        anthropic.py
        gemini.py           # Includes Gemini generation-specific variants
        llama.py
    openai.py               # Existing model compatibility wrappers
    bedrock.py
    ...
```

For example, import implementations from
`archytas.models.providers.bedrock.BedrockProvider` and
`archytas.models.families.anthropic.AnthropicModelFamily`. Existing top-level
model imports, family package exports, and the `archytas.models.provider`
interface import remain supported. Legacy OpenAI and Gemini reasoning settings
remain in their top-level modules to preserve existing overrides.

## Select a family in runtime configuration

Prefix the configured model name with a registered family slug and a colon:

```bash
# These variables are examples for applications that already read them.
# Archytas itself does not load LLM_SERVICE_MODEL from the environment.
export LLM_SERVICE_MODEL="openai:us.openai.gpt-6.1-terra"
# Or select Claude behavior while keeping the same Bedrock provider:
export LLM_SERVICE_MODEL="anthropic:us.anthropic.claude-opus-5-5"
```

```python
import os

from archytas.models.bedrock import BedrockModel

model = BedrockModel({"model_name": os.environ["LLM_SERVICE_MODEL"]})
```

The family prefix is removed before the provider sees the name. It does not
select the provider or change authentication. The built-in slugs are `generic`,
`openai`, `anthropic`, `gemini`, `gemini2`, `gemini3`, and `llama`.

Only registered prefixes match. Names such as `llama3:latest`,
`us.anthropic.claude-sonnet:0`, and Bedrock ARNs remain unchanged. Unknown prefixes
are also passed through unchanged. A recognized prefix without a model name is
an error.

Without a prefix, the wrapper's default family is used; Archytas does not guess a
family from arbitrary model names. Defaults are:

| Model wrapper | Default family |
| --- | --- |
| OpenAI, Azure OpenAI, OpenRouter | OpenAI |
| Anthropic | Anthropic |
| Gemini | Gemini |
| Ollama | Llama |
| Groq | Legacy Groq message/instruction policy |
| Bedrock | OpenAI |

Bedrock's default model is `us.openai.gpt-6.1-terra`. Existing Claude deployments
on Bedrock should explicitly select the `anthropic:` family. Other defaults and
existing configuration fields are unchanged.

## Provider authentication and API settings

Anthropic, Gemini, and Groq resolve API keys from the `api_key` constructor
keyword, then `config.api_key`, then `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or
`GROQ_API_KEY`, respectively. Empty values fall through to the next source.
Configured keys are not written to those environment variables.

Bedrock accepts a credentials profile or explicit AWS credentials. Constructor
credentials take precedence over configured credentials, which take precedence
over environment credentials. Access and secret keys must be supplied together
when passed as constructor keywords. A complete explicit pair does not inherit
an environment session token; supply `aws_session_token` with that pair if needed.

Azure OpenAI accepts an optional `api_version` in the configuration or as a
constructor keyword. The keyword takes precedence; the default remains
`2024-10-21`:

```python
from archytas.models.azure import AzureOpenAIModel

model = AzureOpenAIModel({
    "model_name": "my-deployment",
    "endpoint": "https://example.openai.azure.com",
    "api_version": "2024-10-21",
})
```

## Select components in Python

Optional constructor keywords accept component classes or instances:

```python
from archytas.models.bedrock import BedrockModel
from archytas.models.families import AnthropicModelFamily

model = BedrockModel(
    {"model_name": "us.anthropic.claude-opus-5-5"},
    model_family=AnthropicModelFamily,
)
```

A registered prefix in `model_name` takes precedence over the constructor's
`model_family`. The `provider` keyword can likewise replace the provider. A
provider and family can also be composed directly with
`BaseArchytasModel(config, provider=..., model_family=...)`.

Composition does not imply that a service hosts a particular model or accepts
another service's API options. For example, Claude served through OpenRouter
still uses OpenAI-compatible response parsing, not Anthropic SDK response
parsing. Reasoning parameter encoding is supplied only by providers that
implement it; choosing a family does not send its native API options to every
provider.

## Define a versioned family

Register a subclass with a new, stable slug when behavior changes:

```python
from archytas.models import OpenAIModelFamily, register_model_family


@register_model_family
class OpenAINextModelFamily(OpenAIModelFamily):
    slug = "openai-next"

    def prepare_invoke_kwargs(self, model, kwargs: dict) -> dict:
        # Example: this generation no longer accepts temperature.
        kwargs.pop("temperature", None)
        return kwargs
```

After the module is imported, `openai-next:some-model` selects that class.
Registration rejects a duplicate slug assigned to another class. Use a new slug
rather than replacing the meaning of a deployed version's slug. The initial
`gemini2` and `gemini3` families inherit the existing Gemini policies and provide
separate extension points for generation-specific changes.

## Processing and metadata

Outgoing messages pass through model-specific content filtering, family message
processing, then provider API processing. Stored conversation messages are not
rewritten by these transformations. Prompt cache markers remain provider API
behavior; Bedrock adds Claude markers only for the Anthropic family.

Response content is parsed by the provider, then the family can interpret the
normalized `AgentResponse`. The `_rectify_result` hook separately normalizes raw
messages before they enter history. Model-specific signed content is retained
only for the same provider type, family type, and model name. It is still omitted
from serialized history.

Context limits use the active LangChain model's profile first, followed by
provider discovery, family fallbacks, and finally provider conservative
fallbacks. An active model's profile is not used for another model name.
Temperature requires both an explicit value and profile-confirmed support; a
family can impose additional restrictions. Providers encode supported values
into their API-specific fields. Families return `ReasoningSettings` preferences
(effort, summary, budget, and thought visibility), which providers translate into
their own reasoning parameters.

Synchronous and asynchronous invocation retain their existing differences.
Summarization still accesses the underlying unbound `_model` client directly;
this refactor does not change that path.
