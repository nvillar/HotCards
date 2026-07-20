# Local model defaults decision

**Decision date:** 2026-07-19
**Status:** provisional defaults; human quality review remains required

## Ollama default: `qwen3.5:9b` (provisional)

The final indexed hotspot comparison used the production 4/2/12
interaction/component/point limits, 1024 output tokens, 8192 context tokens,
and a 120 second timeout. All three candidates (`qwen3.5:4b`,
`qwen3.5:9b`, and `qwen3.6:35b`) achieved 100% structured success, no
repetition or limit hits, and 100% geometry validity after clamping.
Pre-cleanup raw validity was 75% for 4B and 9B and 100% for 35B.

All candidates scored 0 destination-token accuracy in that run. Consequently,
35B's stronger raw geometry does not establish greater end-user usefulness,
and none is safe for automatic destination acceptance. The isolated suite initially favored 4B as the smallest candidate meeting the
same structured, repetition, limit, and post-cleanup geometry gates. The
end-to-end run changed that decision: 4B invented two additional interactions
and reached the four-interaction safety limit, while 9B and 35B each returned
the two authored actions. The 35B polygons were tighter, but 9B produced
editable valid geometry at a smaller model size and did not lose any
destination capability relative to 35B. Use 9B as the smallest reliably useful
end-to-end default. Destinations must remain visibly reviewable and editable.

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
  cleanup, explicit failure states, and mandatory destination review.
- **Authoring UI for review/edit/apply: GO**; it is needed to resolve the known
  destination weakness and collect the human rubric.
- **Automatic hotspot acceptance or end-user quality claim: NO-GO** until
  destination accuracy improves and broader author rubrics are scored.
