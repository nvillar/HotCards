# End-to-end evaluation cases

Versioned author inputs and human rubrics for the fixed-axis end-to-end suite.
The suite derives a prompt, generates an image with `flux2-klein-4b`, and
remaps a seeded set of existing hotspots onto that image with each selected
Ollama candidate. The model controls only geometry; labels and destinations
remain authoritative. Cases are project-authored and contain their provenance
and reuse terms; no generated run artifact is tracked here.
