"""Small opt-in live checks for Bedrock model-family inference."""
import os

import pytest
from langchain_core.messages import HumanMessage

from archytas.models.bedrock import BedrockModel
from archytas.models.families import AnthropicModelFamily, OpenAIModelFamily
from archytas.models.providers.base import ModelProvider


@pytest.mark.integration
@pytest.mark.parametrize("family", [OpenAIModelFamily, AnthropicModelFamily])
def test_bedrock_inferred_family_live(request, family):
    if os.environ.get("ARCHYTAS_BEDROCK_TESTS_ENABLED") != "1":
        pytest.skip("Bedrock live tests are not enabled")
    if request.config.getoption("--model-provider") not in ("all", "bedrock"):
        pytest.skip("Bedrock is not selected")

    # Use the same configured models as the live suite, but remove an explicit
    # family prefix so Bedrock must infer the family from the provider ID.
    parser = ModelProvider()
    for configured_name in request.config.getoption("--bedrock-model").split(","):
        configured_family, name = parser.parse_prefix(configured_name.strip())
        if configured_family is family or (
            configured_family is None and BedrockModel.DEFAULT_PROVIDER().parse_prefix(name)[0] is family
        ):
            break
    else:
        pytest.skip(f"No {family.__name__} model configured for Bedrock")

    model = BedrockModel({"model_name": name})
    assert type(model.model_family) is family
    response = model.invoke([HumanMessage(content="Reply with the single word hello.")])
    assert model.process_result(response).text.strip()
