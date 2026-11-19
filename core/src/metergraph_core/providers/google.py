"""Native Google model identity conventions."""

import re
from typing import Any

_VERTEX_MODEL = re.compile(
    r'projects/[^/]+/locations/[^/]+/publishers/google/models/([^/]+)'
)


def model_identity(model: Any, channel: str | None) -> tuple[Any, str | None]:
    """A Vertex resource name identifies both the model and its native channel."""
    match = _VERTEX_MODEL.fullmatch(model) if isinstance(model, str) else None
    return (match.group(1), 'google-vertex-ai') if match else (model, channel)
