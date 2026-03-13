from pydantic import BaseModel as PydanticModel, ConfigDict


class ModelConfig(PydanticModel, extra='allow'):
    model_name: str
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    api_key: str | None = None
    summarization_ratio: float | None = None
    summarization_threshold: int | None = None
    summarization_threshold_pct: int | None = None

    # extra fields --
    # max_tokens: int | None = None
    # region: str | None = None
