# Local model defaults decision

**Decision date:** 2026-07-21
**Status:** provisional defaults; human quality review remains required

## Ollama default: `qwen3.5:9b-mlx` (provisional)

The active runtime uses the locally optimized MLX variant. Historical
comparisons selected the 9B parameter class; the final revision-centric
validation below exercised `qwen3.5:9b-mlx` directly.

The revision-centric workflow no longer asks the model to invent hotspot
semantics or destinations. Production Remap sends at most two existing
hotspot labels per call behind opaque `H1`, `H2`, and subsequent tokens. Output
can contain only token-bound polygons or explicit unlocated results; the
application preserves IDs, labels, actions, destinations, ordering, and old
geometry for unmatched output.

Final live validation on 2026-08-22 used
`hotspot-remap-prompt-v1`/`hotspot-remap-schema-v1`. Smoke passed in cold and
warm phases with no warnings. Across two isolated cases, cold and warm runs for
all three candidates (`qwen3.5:4b`, `qwen3.5:9b-mlx`, and `qwen3.6:35b`)
completed 12/12 structured phases with 100% mapping and post-validation geometry
rates. The 9B grid ablation also mapped both subjects successfully. Its
optional thinking ablation exhausted the 1024-token output limit and returned
no structured response; production therefore keeps thinking disabled.

The end-to-end generated-image case succeeded for all three Ollama candidates
without warnings. Contact-sheet review found editable polygons attached to the
requested gate and chest subjects. The isolated contact sheet likewise showed
the requested cabinet/window and gate/chest subjects across models, although exact
polygon tightness still varies. Retain 9B as the development default: it remains
the middle resource point, the MLX build is the locally optimized runtime, and
the bounded Remap contract removes the previous interaction-invention and
destination-selection risks.

### Coordinate-grounding follow-up

A focused 2026-08-22 experiment used the four authored hotspots on New Worlds,
Map revision 5 as reference geometry for repeated 9B MLX trials. Native
1024×768 coordinates regressed from `0.379` to `0.318` mean polygon IoU and
reduced the mean mapping rate to 80%. Explicit origin, axis, and tight-boundary
instructions also regressed from `0.385` to `0.189` IoU. Production therefore
retains the concise prompt and normalized 0–1000 coordinate space.

Two-hotspot batches produced the only improvement: `0.390` mean polygon IoU
versus `0.379` for four-at-once and `0.323` for one-at-a-time, with 100%
mapping and no failures in five trials. Mean wall time was effectively
unchanged (`11.10s` versus `11.05s`), although prompt tokens increased from
1,314 to 2,426. Production uses batches of two while preserving one atomic
application and safe Undo across all batches. The experiment is intentionally
limited to one authored revision, so broader geometry-quality claims remain
out of scope.

## MFLUX default: `flux2-klein-4b` (provisional)

The controlled image-model axis held the render prompt fixed. FLUX.2 Klein 4B
inference was approximately 11.2–11.6 seconds versus 30.7–32.1 seconds for 9B;
model load was under approximately 0.8 seconds for both. This supports 4B on
latency and resource tradeoffs without confounding the Ollama axis.

The final live revision-centric image suite completed all eight cold/warm
renders for both tracked cases and both MFLUX candidates without warnings.
Manual contact-sheet review found that both variants preserved both scenes,
styles, and the required spatially distinct interactive subjects without
obvious text or UI artifacts. The 9B images changed composition and detail but
did not show a consistent hotspot-suitability gain. Use 4B as the development
default: the observed quality difference does not justify its higher latency
and resource cost. Continue collecting scored author rubrics; this is not a
broad quality-parity claim.

## Go/no-go

- **Controller integration: GO**, with deterministic geometry validation,
  explicit failure states, and stale-result suppression.
- **Apply-first Remap with safe Undo: GO** because model output can alter only
  geometry for existing hotspots and unmatched geometry is preserved.
- **Broad end-user geometry-quality claim: NO-GO** until more diverse author
  rubrics are scored.
