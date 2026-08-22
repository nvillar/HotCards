# Hotspot evaluation cases

The frozen fixtures are deterministic HyperGen project artwork made from
simple geometric shapes. They have no external source material or reuse
restrictions. Case files define existing hotspot labels and request-local tokens
for geometry-only Remap. Geometry quality and edit cost remain human rubric
inputs.

`contracts/remap-v1.json` records the production Remap contract, safety
boundary, selected limits, and model recommendation. Deterministic malformed,
duplicate, unknown-token, unlocated, batching, and clamping regressions live in
the ordinary test suite rather than frozen model-response files.

- `../smoke/courtyard.png` SHA-256:
  `4ddf55edf2b938c854e4e2e196e261eba845199f7571f5ceacceb87bfebe95b1`
- `workshop.png` SHA-256:
  `bcf9c548ca271b7e4d67175b4c4971c8157537912ab22b29762f15c00d1df29b`
