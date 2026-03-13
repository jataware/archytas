"""Backward-compatibility shim.

This module re-exports the Model class as BaseArchytasModel so that existing
consumer code (agent.py, react.py, chat_history.py) continues to work.
All new code should import from archytas.models directly.
"""

import os

from .model import Model as BaseArchytasModel
from .config import ModelConfig
from .tool_convert import final_answer, fail_task, convert_tools


class EnvironmentAuth:
    env_settings: dict[str, str]

    def __init__(self, **env_settings: dict[str, str]) -> None:
        for key, value in env_settings.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                raise ValueError("EnvironmentAuth variables names and values must be strings.")
        self.env_settings = env_settings

    def apply(self):
        for env_name, env_value in self.env_settings.items():
            os.environ.setdefault(env_name, env_value)


def set_env_auth(**env_settings: dict[str, str]) -> None:
    for key, value in env_settings.items():
        if not (isinstance(key, str) and isinstance(value, str)):
            raise ValueError("EnvironmentAuth variables names and values must be strings.")
    for env_name, env_value in env_settings.items():
        os.environ.setdefault(env_name, env_value)


__all__ = [
    "BaseArchytasModel",
    "ModelConfig",
    "EnvironmentAuth",
    "set_env_auth",
    "final_answer",
    "fail_task",
    "convert_tools",
]
