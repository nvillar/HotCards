# Local model defaults decision

**Decision date:** 2026-07-21
**Status:** provisional defaults; human quality review remains required

## Ollama default: `qwen3.5:9b-mlx` (provisional)

The active runtime uses the locally optimized MLX variant. The selection
evidence below was gathered with the base `qwen3.5:9b` tag and remains the
quality rationale for choosing the 9B parameter class.

The MLX variant completed final live validation on 2026-08-22 with the
production v3 prompt and v2 schema. Smoke passed in cold and warm phases. In
the isolated hotspot suite, it achieved 4/4 structured successes, 8/8 correct
destinations, valid raw geometry, no repetition, and no production limit hits.
The end-to-end case added 2/2 correct destinations with valid editable geometry.
Its optional thinking ablation reached the token limit; production keeps
thinking disabled. Contact-sheet review confirmed that its gate and chest
polygons were attached to the intended visible subjects.

The final indexed hotspot comparison used the production 4/2/12
interaction/component/point limits, 1024 output tokens, 8192 context tokens,
and a 120 second timeout. All three candidates (`qwen3.5:4b`,
`qwen3.5:9b`, and `qwen3.6:35b`) achieved 100% structured success, no
repetition or limit hits, and 100% geometry validity after clamping.
Pre-cleanup raw validity was 75% for 4B and 9B and 100% for 35B.

The original object-union destination contract produced 0 destination-token
accuracy. A v2 contract replaced it with one required `destination_token`
constrained to the exact request tokens plus `UNRESOLVED`. In the subsequent
live multimodal comparison, every model resolved all expected destinations on
both cases in cold and warm phases: 24/24 correct. Opaque C1/C2 tokens also
matched descriptive slugs in the focused ablation, so the simpler opaque
request-local identifiers remain appropriate. The v3 prompt now states the
exact JSON nesting because the MLX variant does not reliably honor Ollama's
schema transport on its own; Pydantic validation remains strict.

The isolated suite initially favored 4B as the smallest candidate meeting the
same structured, repetition, limit, and post-cleanup geometry gates. The
end-to-end run changed that decision: 4B invented two additional interactions
and reached the four-interaction safety limit, while 9B and 35B each returned
the two authored actions. The 35B polygons were tighter, but 9B produced
editable valid geometry at a smaller model size and did not lose any
destination capability relative to 35B. Use 9B as the smallest reliably useful
end-to-end default. Destination choices remain visibly reviewable because the
corrected live evidence still covers only two cases.

## MFLUX default: `flux2-klein-4b` (provisional)

The controlled image-model axis held the render prompt fixed. FLUX.2 Klein 4B
inference was approximately 11.2–11.6 seconds versus 30.7–32.1 seconds for 9B;
model load was under approximately 0.8 seconds for both. This supports 4B on
latency and resource tradeoffs without confounding the Ollama axis.

Manual contact-sheet review found that both variants preserved both scenes,
styles, and the required spatially distinct interactive subjects without
obvious text or UI artifacts. The 9B images changed some composition and
detail, but did not show a consistent hotspot-suitability gain. Use 4B as the
development default: the observed quality difference does not justify roughly
three times the inference latency. Continue collecting scored author rubrics;
this is not a broad quality-parity claim.

## Go/no-go

- **Controller integration: GO**, retaining raw output, deterministic geometry
  cleanup, explicit failure states, and visible destination review.
- **Authoring UI for review/edit/apply: GO**; it remains necessary to collect
  the human rubric and correct occasional geometry or interaction errors.
- **Automatic hotspot acceptance or end-user quality claim: NO-GO** until
  broader author rubrics are scored.
