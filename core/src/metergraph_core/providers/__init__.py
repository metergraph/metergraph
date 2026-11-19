"""Resolve provider-native model identities before catalog lookup."""

from typing import Any

from ..catalog import direct_channel_for_provider, normalize_provider
from . import google


def model_identity(provider: Any, model: Any) -> tuple[Any, str | None]:
    channel = direct_channel_for_provider(provider)
    if normalize_provider(provider or '') in {'google', 'vertex-ai'}:
        return google.model_identity(model, channel)
    return model, channel
