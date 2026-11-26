"""Built-in models. Importing this package registers them.

`awsrcf` needs a JVM and the AWS jars at use time, not at import."""

from . import awsrcf, rcf, seasonal, static, toto  # noqa: F401  (registration side effects)
from .base import Model, ModelOutput, available_models, create_model, register_model

__all__ = ["Model", "ModelOutput", "available_models", "create_model", "register_model"]
