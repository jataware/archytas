"""Provider implementations live in their service-specific modules.

Only the shared interface is imported here, so importing it does not load all
provider SDKs.
"""
from .base import ModelProvider

__all__ = ["ModelProvider"]
