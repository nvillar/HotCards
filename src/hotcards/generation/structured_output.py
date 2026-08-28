"""Narrow normalization for structured model-response transport wrappers."""

from __future__ import annotations

import re

_JSON_CODE_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def structured_json_content(content: str) -> str:
    """Remove one standalone JSON fence without accepting surrounding prose."""
    stripped = content.strip()
    match = _JSON_CODE_FENCE.fullmatch(stripped)
    return match.group("body").strip() if match is not None else stripped


__all__ = ["structured_json_content"]
