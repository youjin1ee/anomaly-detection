"""Built-in models. Importing this package registers them."""

from . import rcf, seasonal, static, tspulse  # noqa: F401  (registration side effects)
from .base import Model, ModelOutput, available_models, create_model, register_model

__all__ = ["Model", "ModelOutput", "available_models", "create_model", "register_model"]
