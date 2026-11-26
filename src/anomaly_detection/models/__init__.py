"""Built-in models. Importing this package registers them.

`awsrcf` needs a JVM and the AWS jars at use time, not at import."""

from . import (  # noqa: F401  (registration side effects)
    awsrcf,
    rcf,
    seasonal,
    static,
    subpca,
    toto,
    tspulse,
)
from .base import Model, ModelOutput, available_models, create_model, register_model

__all__ = ["Model", "ModelOutput", "available_models", "create_model", "register_model"]
