"""Shared strict contracts for version-controlled evaluation inputs."""

from typing import Annotated

from pydantic import StringConstraints

SafeCaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
