"""Role-safe text profiles extracted from reference generation provenance."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    ReferenceRole,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime

REFERENCE_PROFILE_PROMPT_VERSION = "reference-profile-v1"


class SubjectReferenceProfile(DomainModel):
    """Stable identity and appearance traits for one Subject reference."""

    identity: str | None = None
    appearance: str | None = None
    body: str | None = None
    clothing: str | None = None
    materials: str | None = None


class StyleReferenceProfile(DomainModel):
    """Reusable rendering traits for one Style reference."""

    medium: str | None = None
    linework: str | None = None
    texture: str | None = None
    palette: str | None = None
    shading: str | None = None
    rendering: str | None = None


class SettingReferenceProfile(DomainModel):
    """Stable place traits separated from transient observed conditions."""

    environment: str | None = None
    architecture: str | None = None
    materials: str | None = None
    terrain: str | None = None
    spatial_character: str | None = None
    observed_weather: str | None = None
    observed_time: str | None = None
    observed_season: str | None = None


ReferenceProfile = SubjectReferenceProfile | StyleReferenceProfile | SettingReferenceProfile


class ReferenceProfileRequest(DomainModel):
    """One role-specific profile extraction request."""

    role: ReferenceRole
    source_description: NonEmptyString
    prompt_version: NonEmptyString = REFERENCE_PROFILE_PROMPT_VERSION


class ReferenceProfileResult(DomainModel):
    """Validated profile and deterministic capsule for enrichment."""

    role: ReferenceRole
    profile: ReferenceProfile
    capsule: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString


_SUBJECT_RULES = """\
Keep only stable identity and recognizable appearance.
- identity is the subject's stable type, species, name, or role.
- appearance is face, hair, markings, and distinctive features.
- body is anatomy, build, proportions, and limb structure only. Never include posture or
  pose words such as seated, standing, kneeling, running, dancing, facing, or profile.
- clothing is worn garments and accessories.
- materials contains only explicitly stated physical materials. Never infer generic
  materials; colors such as copper hair belong in appearance, not materials.
Exclude action, pose, expression, held objects, setting, background, camera, composition,
lighting, time, weather, and rendering style.
"""

_STYLE_RULES = """\
Keep only reusable medium, linework, texture, palette, shading technique, and rendering
treatment. Every retained phrase must work unchanged on an unrelated portrait, vehicle,
landscape, or interior. Exclude every subject, object, place, architecture, environment,
action, viewpoint, composition, time, weather, depicted light source, and narrative detail.
"""

_SETTING_RULES = """\
Keep only stable place identity, architecture, intrinsic materials, terrain, and spatial
character.
- environment is the place type stripped of temporary state. Remove words such as occupied,
  empty, crowded, abandoned, lit, dark, snowy, rainy, daytime, or nighttime unless they name
  a permanent place type.
- architecture is permanent built form.
- materials are explicitly stated intrinsic construction materials.
- terrain is stable landform and vegetation; normalize actions such as swaying kelp to kelp.
- spatial_character is stable scale, enclosure, and connectivity only.
- observed_weather, observed_time, and observed_season record explicit transient conditions
  for provenance, but these fields are not Setting traits and will not be applied.
Exclude every occupant, action, portable object, viewpoint, camera, composition, temporary
illumination, and rendering style from the reusable Setting fields.
"""

_PROFILE_TYPES = {
    ReferenceRole.SUBJECT: SubjectReferenceProfile,
    ReferenceRole.STYLE: StyleReferenceProfile,
    ReferenceRole.SETTING: SettingReferenceProfile,
}

_PROFILE_RULES = {
    ReferenceRole.SUBJECT: _SUBJECT_RULES,
    ReferenceRole.STYLE: _STYLE_RULES,
    ReferenceRole.SETTING: _SETTING_RULES,
}

_CAPSULE_FIELDS = {
    ReferenceRole.SUBJECT: (
        "identity",
        "appearance",
        "body",
        "clothing",
        "materials",
    ),
    ReferenceRole.STYLE: (
        "medium",
        "linework",
        "texture",
        "palette",
        "shading",
        "rendering",
    ),
    ReferenceRole.SETTING: (
        "environment",
        "architecture",
        "materials",
        "terrain",
        "spatial_character",
    ),
}


def build_reference_profile_prompt(request: ReferenceProfileRequest) -> str:
    """Build the strict role-specific extraction prompt proven by live trials."""
    profile_type = _PROFILE_TYPES[request.role]
    keys = ", ".join(profile_type.model_fields)
    return f"""\
Extract a role-safe {request.role.value.upper()} profile from the source Description.

{_PROFILE_RULES[request.role]}
Every reusable phrase must remain true when all excluded details are replaced.

OUTPUT CONTRACT
- Return exactly one JSON object and nothing else.
- The keys must be exactly: {keys}.
- Every value must be either one concise JSON string or null.
- Combine multiple values for one field into one comma-separated string.
- Do not emit arrays, nested objects, Markdown, renamed keys, or additional keys.

Prompt contract: {request.prompt_version}
Source Description:
{request.source_description}
"""


def profile_capsule(
    role: ReferenceRole,
    profile: ReferenceProfile,
) -> str:
    """Serialize only reusable fields in deterministic order."""
    values = [
        value.strip()
        for field in _CAPSULE_FIELDS[role]
        if isinstance((value := getattr(profile, field)), str) and value.strip()
    ]
    if not values:
        raise ValueError(f"{role.value} reference profile contains no reusable traits")
    return "; ".join(values)


class OllamaReferenceProfiler:
    """Extract strict role profiles through the shared Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def extract(
        self,
        request: ReferenceProfileRequest,
    ) -> ReferenceProfileResult:
        profile_type = _PROFILE_TYPES[request.role]
        call = self._runtime.chat_structured(
            prompt=build_reference_profile_prompt(request),
            schema=profile_type.model_json_schema(),
        )
        try:
            profile = profile_type.model_validate_json(call.content)
            capsule = profile_capsule(request.role, profile)
        except (ValidationError, ValueError) as error:
            raise ModelResponseError(
                f"Ollama returned an invalid {request.role.value} reference "
                f"profile for {request.prompt_version}: {error}",
                raw_response=call.content,
                response_metadata={
                    "elapsed_seconds": call.elapsed_seconds,
                    "total_duration_ns": call.total_duration_ns,
                    "load_duration_ns": call.load_duration_ns,
                    "prompt_eval_count": call.prompt_eval_count,
                    "eval_count": call.eval_count,
                    "done_reason": call.done_reason,
                },
            ) from error
        return ReferenceProfileResult(
            role=request.role,
            profile=profile,
            capsule=capsule,
            model_identifier=self._runtime.settings.model,
            prompt_version=request.prompt_version,
        )


def reference_capsules(
    profiles: Sequence[ReferenceProfileResult],
) -> tuple[tuple[ReferenceRole, str], ...]:
    """Return role capsules in deterministic role order."""
    by_role = {profile.role: profile.capsule for profile in profiles}
    return tuple((role, by_role[role]) for role in ReferenceRole if role in by_role)


__all__ = [
    "OllamaReferenceProfiler",
    "REFERENCE_PROFILE_PROMPT_VERSION",
    "ReferenceProfile",
    "ReferenceProfileRequest",
    "ReferenceProfileResult",
    "SettingReferenceProfile",
    "StyleReferenceProfile",
    "SubjectReferenceProfile",
    "build_reference_profile_prompt",
    "profile_capsule",
    "reference_capsules",
]
