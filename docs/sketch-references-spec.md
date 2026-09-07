# Sketch References, Version-Local Edit Drafts, and Evolve Removal

**Repository:** `nvillar/HotCards`  
**Specification date:** September 7, 2026  
**Status:** Sketch-reference work parked; reduced non-sketch scope adopted

**Intended file:** `docs/sketch-references-spec.md`

> **Implementation scope:** Do not implement the sketch editor, sketch catalog,
> sketch-backed Generate references, or Edit sketch attachments from this
> document. The adopted work is limited to removing executable Evolve while
> retaining historical Refine provenance, and making text-only Edit drafts
> revision-local with the lifecycle and durability semantics specified here.
> Evolve removal is the current implementation phase; text-only revision-local
> drafts follow in later phases. Except for those two areas and the one-off
> stack conversion they require, the sections below describe the parked,
> non-normative sketch proposal.
> The sketch feasibility findings are retained below for future reconsideration.

---

## 1. Purpose

Simplify image authoring into two operations:

- **Generate:** create an image from a Description, optional Style, and optional references.
- **Edit:** change the current image using a new Edit Instruction and an optional sketch.

Remove **Evolve** as an executable authoring operation.

Introduce a small raster drawing editor whose sketches can serve as Generate references or Edit attachments. Make unfinished Edit instructions and their sketches belong to individual card versions, rather than shared inspector controls.

The feature must preserve Escape To Earth's existing content and accepted image history through the one-off conversion in Section 15, while preserving durable image application and protection against stale asynchronous results. The application will support only the new schema, not legacy-schema loading or runtime migration.

### 1.1 Core principles

1. **The current image is the target of Edit.**
2. **Description is an input to Generate, not a synchronized description of the current image.**
3. **An Edit draft is an instruction plus an optional sketch.**
4. **Each card version owns its own Edit draft.**
5. **Sketches are ordinary visual references, not masks.**
6. **The application does not assign meanings to sketch colors.**
7. **Changing the base image does not invalidate a sketch or require a warning.**
8. **History recall restores inputs, not a historical image state.**
9. **Inference completion is not equivalent to durable application.**
10. **Historical sketches must remain immutable and recallable while their history is retained.**

---

## 2. Original Sketch Scope (Parked)

### 2.1 Originally included

Everything in this subsection is parked except Evolve removal, historical
Refine preservation, text-only revision-local Edit drafts, their non-sketch
lifecycle/Undo/durability behavior, and the final stack conversion.

- Removal of Evolve UI and executable workflow.
- Historical Refine/Evolve provenance within the new schema.
- A final, one-off conversion of Escape To Earth after making a complete schema 13 backup.
- A shared raster sketch editor.
- Sketches in either of Generate’s two reference slots.
- One optional sketch in Edit.
- Transparent editable drawings and white-background model inputs.
- Version-local, persisted Edit drafts.
- Instruction/sketch consumption and recall as a pair.
- Copy-on-write sketch editing.
- Asset retention, duplication, Save As, and cleanup.
- Undo/Redo integration.
- Stale-result and cancellation safeguards.
- Automated tests and explicit live model evaluations.

### 2.2 Excluded

- Legacy-schema readers, compatibility branches, and runtime migration logic.
- A maintained migration framework or support for converting other stacks.
- Image import.
- Masking or inpainting controls.
- Automatic interpretation of colors.
- Automatic prompt rewriting for sketches.
- Automatic alignment or repositioning of drawings.
- Description synchronization after Edit.
- Automatic replay of Edit History.
- New Evolve-like strength or source-similarity controls.
- Vector/object editing.
- Text tools, crop, flood fill, selection/move, or multilayer authoring.
- A general-purpose drawing application.
- New zoom or pan controls on the main card canvas.

Zoom and pan in this specification apply only to the sketch editor.

---

## 3. Final Authoring Model

| Operation | Required text | Image inputs | Style behavior |
|---|---|---|---|
| Generate | Nonempty Description | Zero, one, or two ordered references | Existing Generate composition |
| Edit | Nonempty Edit Instruction | Current image, plus zero or one sketch | Existing Edit visual-continuity behavior |

### 3.1 Generate

Generate uses:

- Current Description.
- Selected Style, if any.
- Up to two ordered references.
- Selected Generate output size.

References may be:

- Card references.
- Sketches.
- One of each.
- Two sketches.

Generate does not automatically use:

- The current image.
- The active Edit draft.
- An Edit attachment.
- Accumulated Edit History.

A successful Generate starts a new image lineage, as it does today.

### 3.2 Edit

Edit uses:

- The readable current background.
- The authored Edit Instruction.
- One optional sketch.
- Existing Edit output-size behavior.
- Existing Style-continuity prompt behavior.

Edit does not send:

- Description.
- Generate references.
- Earlier accepted instructions as accumulated editing guidance.

Historical lineage remains available for provenance and recall, not prompt reconstruction.

### 3.3 No implicit transfer between modes

- Generate references never become Edit attachments automatically.
- Edit sketches never become Generate references automatically.
- Switching tabs does not consume or transfer a sketch.
- Generating a new background does not clear an unfinished Edit draft.

---

## 4. Evolve Removal

### 4.1 Remove from active authoring

Remove:

- Evolve inspector tab.
- Evolve action and controls.
- Source Similarity selector and descriptions.
- Evolve-specific resolution selector.
- Evolve-specific progress and error UI.
- Refine request construction and executable workflow entry points.
- Refine-only worker operations where no longer needed.
- Evolve prompt composition that reconciles Description and accepted Edit instructions.
- Evolve-only evaluation commands and executable cases.

The inspector tab order becomes:

1. Generate
2. Edit
3. Hotspots

Do not rely on old numeric tab indices after removal.

### 4.2 Keep historical provenance support

Retain the data types and validation needed to read historical facts within the new schema:

- `refine` provenance.
- Refine source snapshots.
- Refine transformation and strength facts.
- Refine output-size facts.
- Inherited accepted Edit lineage.
- Duplicates whose original provenance is Refine.

Existing Evolve-created backgrounds must remain:

- Displayable.
- Selectable as active backgrounds.
- Duplicable.
- Usable as card references.
- Usable as Edit sources.
- Saveable without changing historical facts.

Do not relabel historical Evolve operations as Generate or Edit.

This is historical operation support, not support for loading schema 13 or older bundles. Section 15 defines the one-off conversion of the only existing stack in scope.

### 4.3 Preserve shared functionality

Do not remove regular `Flux2Klein` support merely because Evolve used it. Generate without references still requires the existing plain-generation path.

Do not remove shared:

- Output-dimension validation.
- Source snapshot handling.
- Model invocation serialization.
- Image storage transactions.
- Provenance flattening.
- Edit lineage extraction.

Separate historical provenance support from executable operation support and legacy-schema loading.

### 4.4 Description after removal

Description remains author-controlled Generate input.

After an Edit:

- Do not rewrite Description.
- Do not append instructions to Description.
- Do not warn that Description and image differ.
- Do not treat their divergence as invalid state.

---

## 5. Generate Reference UX

### 5.1 Slot behavior

Keep one References section with a maximum of two occupied slots.

Maintain visible positional labels:

- `1.`
- `2.`

User guidance continues to refer to **image 1** and **image 2**.

Each slot can contain either a card reference or a sketch. Sketches do not receive special priority or forced ordering.

Removing slot 1 promotes slot 2. Update labels without rewriting Description.

### 5.2 Creating a sketch

An empty slot offers:

- Choose Card.
- Draw Sketch.

**Implementation default:** expose these through a compact add-menu rather than adding another permanent panel.

Choosing Draw Sketch opens a blank drawing editor:

- White display background.
- No current card image.
- No other reference image underneath.

A new sketch does not occupy the slot until the user selects **Use Sketch**.

Cancel leaves the slot unchanged.

### 5.3 Attached sketches

An attached sketch displays:

- Its slot number.
- A white-background thumbnail.
- An accessible label identifying it as a sketch.
- The existing compact remove control.

Activating the thumbnail reopens the drawing for editing.

Replacing a card reference with a sketch, or a sketch with a card reference, is explicit and undoable.

### 5.4 Persistence and consumption

Generate references are saved authoring inputs, not one-shot drafts.

A successful Generate:

- Does not clear its references.
- Does not clear sketches from its slots.
- Records immutable reference snapshots in provenance.

Editing a saved sketch later must not alter a previously generated image’s recorded inputs.

### 5.5 Card-reference rules

Existing card-reference rules remain:

- No self-reference.
- No duplicate card reference within the two slots.
- Existing unresolved-reference behavior.
- Existing eligibility rules.

Two separately authored sketches may have identical pixels. Do not reject sketches based on pixel similarity.

---

## 6. Edit Attachment UX

### 6.1 Layout

The Edit tab contains:

1. Edit Instruction.
2. Optional sketch attachment.
3. Resolution.
4. Edit action.
5. Edit History.

The current canvas/header remains the implicit Edit target.

Do not add a separate current-image attachment slot.

### 6.2 Attachment behavior

Edit permits exactly one sketch.

- Add opens the drawing editor over the current image.
- Activate the thumbnail to reopen it.
- Remove detaches the sketch without clearing the instruction.
- Editing the instruction does not detach the sketch.
- Canceling the drawing editor leaves the attached sketch unchanged.
- Use Sketch replaces the attachment as one draft action.

A sketch alone is insufficient to submit Edit. The instruction must remain nonempty.

### 6.3 Availability

Creating or reopening an Edit sketch requires a readable current image.

If the image is unavailable:

- Preserve the instruction and attached sketch.
- Keep the thumbnail visible.
- Disable drawing-editor entry and Edit submission with the normal disabled-state reason.

Do not display a warning merely because a different image has replaced the one previously shown underneath the sketch.

---

## 7. Sketch Representation

### 7.1 Canonical editable representation

Persist the drawing as an RGBA raster image.

- Untouched pixels are transparent.
- Painted pixels carry color and alpha.
- Antialiased edges may contain partial alpha.
- The base image is not included.
- White paint remains distinct from transparent pixels.

Use an immutable PNG asset for each committed drawing version.

### 7.2 Model-input representation

Produce a separate RGB rendering by compositing the drawing over solid white.

Conceptually:

```text
Model input = transparent drawing over white
```

For Edit, the base remains a separate model input:

```text
Edit input 1 = current image
Edit input 2 = drawing flattened over white, if attached
```

Never send the editor’s combined base-plus-drawing preview as the sketch input.

### 7.3 Thumbnail representation

Display the drawing over white for attachment thumbnails.

This keeps the thumbnail representative of the actual sketch image sent to the model.

A thumbnail must not include:

- The base image.
- Selection handles.
- Cursor previews.
- Grid overlays.
- Editor chrome.

### 7.4 No white-to-transparent conversion

Never recover editable transparency by removing white from the flattened export.

That would destroy intentional white paint and change historical drawings.

Always retain the original transparent representation for reopening and recall.

### 7.5 Empty drawings

**Implementation default:** a drawing is empty only when every pixel has zero alpha.

Therefore:

- A fully erased drawing is empty.
- A white-only drawing is not empty.
- Use Sketch is disabled for an empty drawing.
- Removing an existing attachment uses the attachment’s remove control.

Do not add visual-content heuristics or warnings for low-contrast drawings.

---

## 8. Drawing Editor

### 8.1 Window lifecycle

**Implementation default:** use one resizable modal drawing dialog.

This keeps the editing context stable while a drawing is open and avoids silently transferring an unfinished drawing to another card.

Buttons:

- **Use Sketch**
- **Cancel**

Closing the window or pressing Escape behaves as Cancel.

The dialog owns an uncommitted working copy. It never edits an attached or historical asset in place.

### 8.2 Display layers

Generate editor:

```text
Drawing layer
White display background
```

Edit editor:

```text
Drawing layer
Current base image
```

The base image is read-only.

Do not expose a base-image visibility toggle in this feature.

### 8.3 Tools

Provide:

- Freehand brush.
- Straight line.
- Rectangle.
- Ellipse.
- Eraser.
- Undo.
- Redo.
- Clear.
- Zoom.
- Pan.
- Fit.

Rectangle and ellipse support outline and filled modes.

Use a single active drawing color. No separate fill/stroke colors are required.

### 8.4 Palette and widths

**Implementation defaults:**

| Color | Value |
|---|---|
| Black | `#000000` |
| White | `#FFFFFF` |
| Red | `#E53935` |
| Orange | `#FB8C00` |
| Yellow | `#FDD835` |
| Green | `#43A047` |
| Blue | `#1E88E5` |
| Purple | `#8E24AA` |

Stroke widths at canonical drawing resolution:

- Thin: 3 px.
- Medium: 8 px.
- Thick: 16 px.

Default:

- Brush.
- Black.
- Medium width.
- Outline shape mode.

Colors have no application-defined semantics. Red is not automatically a removal instruction, and green is not automatically an addition instruction.

### 8.5 Raster behavior

- Brush strokes use continuous interpolation.
- Brush and line tools use round caps.
- Eraser restores transparency.
- White paint is ordinary opaque paint.
- Shape previews are transient until pointer release.
- Escape cancels an unfinished gesture.
- Each completed gesture is one editor-local Undo action.
- Clear is one undoable editor-local action.

Editor-local Undo must not trigger document Undo.

### 8.6 Coordinates and resolution

**Implementation default:** use the stack’s existing canonical logical canvas:

| Format | Drawing dimensions |
|---|---|
| Square | 1024 × 1024 |
| Landscape | 1024 × 768 |
| Portrait | 768 × 1024 |
| Widescreen | 1024 × 576 |

Drawing dimensions do not depend on selected output resolution.

Changing output resolution:

- Does not resize the stored drawing.
- Does not alter its relative placement.
- Does not modify its identity.

Map pointer coordinates through the inverse viewport transform. Zooming must not change the actual brush width in drawing pixels.

### 8.7 Base-image fitting

Fit the full current image into the canonical drawing canvas with aspect-preserving scaling.

- No cropping.
- No automatic stretching.
- Center the fitted image.
- Preserve drawing coordinates independently of the fitted image.

For unusual historical aspect ratios, display any surrounding area neutrally. Do not warp or reposition the sketch to match the source.

### 8.8 Export sizing

Use the canonical drawing dimensions for the white-background reference export.

Do not rescale it merely because the requested output tier changes.

Any adapter-required preprocessing must:

- Be deterministic.
- Preserve the full drawing.
- Avoid cropping.
- Avoid introducing sketch-specific semantics.
- Be covered by input-contract tests.

### 8.9 Accessibility

Provide:

- Accessible tool names.
- Keyboard navigation.
- Clearly indicated selected tool/color/width.
- Color names in addition to swatches.
- Standard Undo/Redo shortcuts.
- Accessible Use Sketch and Cancel buttons.

The sketch editor must not capture shortcuts after it closes.

---

## 9. Version-Local Edit Drafts

### 9.1 Ownership

A draft belongs to:

```text
stack / card UUID / revision UUID
```

It does not belong to:

- An inspector widget.
- The currently selected card name.
- A visible revision number.
- A particular background identity.

Visible version numbers are positional; use stable UUIDs internally.

### 9.2 Draft contents

A draft consists of:

- Raw instruction text.
- Optional immutable sketch asset reference.
- A draft identity/generation used for change tracking.

Keep raw text while editing. Trim only at submission, following the existing Edit instruction contract.

**Implementation default:** changing text, replacing/removing a sketch, or recalling history creates a fresh draft generation. A recall counts as new input even when the text and sketch match an earlier draft.

### 9.3 Persistence

Persist unfinished instruction text and committed sketch attachments in the stack bundle.

They survive:

- Card navigation.
- Version navigation.
- Application close and reopen.
- Save As.

Uncommitted changes inside an open sketch editor are excluded until Use Sketch.

Use existing debounced saving for text, with a synchronous flush before lifecycle actions that could otherwise lose it.

A save failure must retain the draft in memory and use existing save-error handling.

### 9.4 Navigation behavior

| Action | Required result |
|---|---|
| A → B | Show B’s draft or empty controls |
| B → A | Restore A’s exact draft |
| Switch versions | Show the destination version’s draft |
| Switch inspector tabs | Preserve the version’s draft |
| Open another stack | Load only that stack’s drafts |
| No selected card | Clear visible controls without deleting saved drafts |

Block widget signals during rendering so loading a draft does not generate another edit action.

### 9.5 Base-image changes

Changing a background leaves the draft untouched.

Do not:

- Warn.
- Flag it as stale.
- Require review.
- Clear the attachment.
- Change its placement.
- Rewrite the instruction.

Reopening the editor displays the same drawing over the current image.

Source-image identity may still be used for request safety and provenance. It must not restrict reuse of the draft.

---

## 10. Submission and Consumption

### 10.1 Request capture

Before dispatch, capture an immutable request containing:

- Project/session identity.
- Card and revision UUIDs.
- Current background identity and verified source snapshot.
- Draft identity.
- Exact submitted instruction.
- Optional sketch identity and immutable input snapshot.
- Effective prompt.
- Style snapshot where applicable.
- Output size.
- Model settings.

Workers must not read mutable widget state.

### 10.2 Successful consumption

Consume a draft only after the result has been definitively applied and durably saved.

Consumption clears:

- The submitted instruction.
- The submitted sketch attachment.

It does not delete retained historical assets.

Clear only when the owning version still contains the exact submitted draft generation.

Never clear:

- Another card’s draft.
- Another version’s draft.
- A newer instruction.
- A newer sketch.
- An explicitly recalled draft, even if its contents are identical.

### 10.3 Atomicity

For a matching draft, persist these as one durable acceptance transition:

1. New background.
2. Accepted Edit provenance and sketch reference.
3. Cleared active draft.

Avoid a crash window in which the new image is saved but the already-consumed request reappears after reopening.

When a newer draft exists, persist the accepted image while preserving that newer draft, provided the request remains eligible under existing stale-result rules.

### 10.4 Failure and cancellation

Retain the originating draft after:

- Validation failure.
- Model failure.
- Cancellation.
- Rejected stale result.
- Storage failure before application.
- Indeterminate durability pending retry.

Do not show normal success actions until durability is confirmed.

Delayed save retry must complete acceptance and consumption at most once.

### 10.5 Input changes during work

Preserve existing cancellation/staleness behavior for request-affecting changes.

Attaching, replacing, or removing a submitted sketch is an input change.

Passive operations are not input changes:

- Rerender.
- Thumbnail generation.
- Viewport zoom.
- Autosave completion.
- Visiting an unchanged draft.

Identity checks remain necessary even when UI controls normally prevent editing during inference.

---

## 11. Edit History

### 11.1 Source of history

Continue deriving visible Edit History from the active background’s canonical accepted Edit lineage, including flattened duplicate provenance.

Do not create an unrelated stack-global history journal.

Consequences:

- Edit appends one accepted entry.
- Generate starts fresh lineage.
- Historical Refine images retain their inherited lineage.
- Another version may expose different history.

### 11.2 Entry content

Each accepted Edit retains:

- Exact accepted authored instruction.
- Exact effective prompt and legacy metadata required by provenance.
- Optional immutable sketch snapshot.
- Enough sketch metadata to reopen the transparent drawing.

Visible history shows:

- Authored instruction.
- A compact sketch thumbnail when present.

Preserve existing chronological ordering, duplicate entries, and text formatting conventions.

Do not expose expanded prompts or Style addenda as the authored instruction.

### 11.3 Recall

Recalling an entry replaces the active version’s complete draft:

```text
instruction + optional sketch
```

Text-only recall removes the active sketch.

Recall does not:

- Submit Edit.
- Restore the old background.
- Navigate to the historical source.
- Restore a historical Style selection.
- Change the output-size selection.
- Warn about base-image changes.

The recalled drawing opens over the current image.

### 11.4 Reversible replacement

History recall must be undoable as a draft replacement.

The previous instruction and sketch must be recoverable together.

**Implementation default:** maintain an application-owned, context-local draft Undo stack separate from image/document Undo. Use an explicit draft replacement action and route shortcuts by focus.

This avoids putting every draft keystroke into the existing image-result history.

### 11.5 Copy-on-write

Recall may initially reference the historical immutable sketch.

Editing it:

- Opens a private working copy.
- Produces a new immutable asset on Use Sketch.
- Updates only the draft.
- Does not modify the historical entry.

Cancel retains the originally recalled attachment.

---

## 12. Undo, Redo, and Create New Version

### 12.1 Undo of an applied Edit

Undo restores the prior image using the existing image-operation boundary.

Restore the consumed instruction/sketch pair only if the owning draft has not been superseded by newer user input.

An intentionally cleared newer draft still counts as newer input. Empty text alone is not sufficient permission to overwrite it.

Restore into the owning version’s state, not whichever controls happen to be visible.

Do not navigate solely to show the restored draft.

### 12.2 Redo

Redo clears only the exact draft automatically restored by that operation’s Undo.

Preserve drafts changed through:

- Typing.
- Sketch editing.
- Sketch removal.
- Explicit clearing.
- History recall.

Bind automatic restoration to the exact image Undo token.

### 12.3 Persisted drafts versus document history

Do not allow restoration of an old document snapshot to overwrite unrelated newer drafts.

The controller/history implementation must merge draft state according to ownership and generation, rather than blindly restoring every historical draft field.

This is especially important once drafts become persisted revision data.

### 12.4 Create New Version

Preserve the existing behavior:

- Restore the original version’s prior accepted authoring state.
- Create and activate a new version containing the accepted result.
- Do not invoke the model again.
- Keep a separate Undo boundary for version creation.

For Edit:

- The already-consumed draft must not reappear merely because the prior complete revision is restored.
- Accepted sketch history follows the result.
- A newer unfinished draft stays with its original version.
- The new result version starts with an empty Edit draft.

Undoing version creation does not restore the submitted Edit draft. Undoing the actual image operation may restore it under the safeguards above.

### 12.5 Keep

Keep dismisses result actions only.

It does not consume the draft again or create another document change.

---

## 13. Duplication, Deletion, and Lifecycle

### 13.1 Duplicate version

Copy:

- Accepted background and provenance.
- Generate Description, Style, references, and output size.
- Hotspots under existing duplication semantics.

Start with an empty Edit draft.

### 13.2 Duplicate card

Preserve existing card duplication behavior.

Additionally:

- Copy saved Generate sketch references semantically.
- Preserve accepted Edit sketch history.
- Do not copy unfinished Edit drafts.

Immutable stack-owned sketch bytes may be shared inside the same bundle. Editing either copy creates a new asset.

Deleting the source card must not invalidate the duplicate’s sketches.

### 13.3 Delete card or version

Deletion removes its active draft from the current document.

Retain its sketches while document Undo/Redo can restore them.

Undo deletion restores the draft and assets.

### 13.4 Save As

Copy all required sketch assets into the destination bundle.

The new bundle must not depend on:

- The original bundle’s filesystem paths.
- Temporary request directories.
- Another card’s removable private assets.

Invalidate or cancel in-flight requests according to the existing Save As lifecycle.

### 13.5 Run mode

- Do not expose sketch authoring in Run.
- Persist/flush committed drafts through existing lifecycle preparation.
- Cancel active image work as today.
- Preserve drafts across entering and leaving Run.

---

## 14. Domain and Storage Design

### 14.1 Typed records

**Implementation default:** introduce typed records equivalent to:

```text
SketchAsset
  id
  relative_path
  width
  height
  content_digest
  format_version

SketchReference
  type = "sketch"
  sketch_asset_id

EditDraft
  generation_id
  instruction
  sketch_asset_id | none

SketchInputSnapshot
  sketch_asset_id
  editable_content_digest
  exported_content_digest
  width
  height
  export_format_version
```

Exact Python names may follow repository conventions.

Use strict discriminated unions, not loosely related nullable fields.

### 14.2 Generate reference union

Extend Generate references to distinguish:

- Existing resolved/unresolved card references.
- Sketch references.

Do not add sketch variants to Hotspot navigation targets.

Generate provenance needs an ordered union of:

- Historical card-image snapshots.
- Sketch input snapshots.

### 14.3 Sketch storage

**Implementation default:**

```text
assets/sketches/<sketch-id>/drawing.png
```

Use stack-owned immutable assets.

Validate:

- Relative path containment.
- Expected PNG format.
- Decoded dimensions.
- Pixel-count and encoded-size limits.
- Alpha-capable editable representation.
- Digest consistency.
- No symlink traversal.
- Owned file identity during cleanup.

Do not persist absolute paths or temporary request paths.

### 14.4 Export retention

Persist the canonical transparent PNG.

White exports and thumbnails may be derived caches, provided:

- Export behavior is versioned and deterministic.
- Accepted provenance records the input digest.
- Rebuilding a cache reproduces the intended pixels.
- A cache is never required to reopen the drawing.

If exact exported bytes become necessary for reproducibility, retain them as separately owned immutable assets rather than changing the canonical drawing.

### 14.5 Asset reachability

Retain sketch assets reachable from:

- Generate reference slots.
- Active Edit drafts.
- Accepted Edit lineage.
- Generate provenance.
- Duplicate provenance.
- Document Undo/Redo.
- Draft replacement Undo/Redo.
- Pending durable transactions.
- Active request snapshots, where applicable.

Historical sketch references are retention roots because recall requires their bytes.

Historical source-image attribution remains informational and does not automatically retain old background files.

### 14.6 Cleanup

Reclaim assets only after all relevant roots release them.

- Never delete by unvalidated path alone.
- Never delete a file whose identity no longer matches the owned asset.
- Do not reclaim an asset still required by a saved manifest.
- Release temporary snapshots after success, cancellation, or failure.
- Preserve rollback safety when asset creation succeeds but manifest persistence fails.

### 14.7 Missing or corrupt sketches

Do not silently convert a sketch-bearing request into a text-only request.

- Missing active attachment: block submission and offer removal/replacement.
- Missing historical attachment: report recall failure without partially overwriting the current draft.
- Corrupt asset during request snapshotting: fail before inference.
- Invalid bundle on open: preserve the currently open document.

---

## 15. New Schema and One-Off Stack Conversion

The inspected baseline uses schema version 13 and strict model validation.

Adding persisted drafts, sketch assets, and reference variants requires an explicit schema change.

**Binding rollout decision:** Escape To Earth is the only existing stack to convert. Make a complete schema 13 backup first, then convert the working stack as the final step of feature delivery. Do not add legacy-schema support or migration logic to the application.

### 15.1 Implementation default

Use schema version 14 if no intervening schema change has occurred.

If another change has already consumed that version, use the next available version and update fixtures accordingly.

The production loader accepts only the new schema. Reject schema 13 and older bundles through normal load-error handling without rewriting them. Do not retain old-schema readers, add automatic upgrades on open, or introduce a migration framework.

Historical Refine and existing legacy Generate provenance remain valid operation records within the new schema. Keeping those facts readable does not require keeping their former bundle schemas readable.

### 15.2 One-off Escape To Earth conversion

Perform this only after implementation, review, automated validation, live model evaluation, and documentation updates are complete:

1. Identify the working Escape To Earth bundle and validate that it is schema 13 using the pre-change baseline tooling.
2. Create and verify a separate, complete schema 13 backup of the bundle, including its manifest and all assets. Leave that backup untouched by conversion.
3. Convert a working copy through manual patching or a one-off external script. This tooling is a delivery artifact, not maintained application code or a reusable migration subsystem.
4. Validate the converted document against the new strict schema, confirm content and asset preservation, and exercise opening and saving with the updated application before replacing the working bundle.

The conversion must preserve these invariants:

- Existing card references retain their meaning and order.
- Revisions receive empty Edit drafts.
- Existing accepted edits receive no sketch attachment.
- The new sketch catalog starts empty.
- Historical Refine facts and inherited accepted Edit lineage remain unchanged, including operations embedded in duplicate provenance.
- Existing image paths, seeds, prompts, and output facts remain intact.
- Cards, versions, IDs, selections, Styles, Keys, Sounds, and hotspots retain their existing meaning.
- All asset files retain their exact bytes, including currently unreferenced files. Do not combine conversion with asset cleanup.

Do not add support for schema 12 or older, convert other stacks, or modify any existing older backups.

### 15.3 Persistence safety

Do not rewrite a bundle merely to inspect it.

Use the normal safe save path for the validated new-schema document. A failed conversion or save must leave the original working bundle recoverable and the schema 13 backup untouched.

Do not reopen and resave the backup with the updated application. Keep source validation and transformation outside production loading and saving.

Older applications are not expected to read the new schema. Do not claim backward compatibility in that direction.

### 15.4 Repository guidance

Implementation must update `AGENTS.md` and `README.md` where their existing rules conflict with:

- Evolve removal.
- Expanded reference types.
- Persisted Edit drafts.
- Sketch-aware history.
- New-schema-only loading and the absence of runtime migration.

This specification is explicitly requested design documentation, not a second maintained progress tracker. Track delivery through issues and pull requests.

---

## 16. Prompt and Model Integration

### 16.1 Generate

Preserve deterministic Description-plus-Style composition.

Pass each reference exactly once, in visible slot order.

Do not insert:

- “This is a sketch.”
- Color interpretation rules.
- Preservation instructions.
- Card-name aliases.
- Hidden reference-role guidance.

### 16.2 Edit

Keep the existing authored-instruction and Style-continuity policy.

The sketch must not alter prompt construction.

Preserve:

- Exact trimmed authored instruction.
- Existing deterministic Style handling.
- Hard token-budget validation.
- No rewriting or truncation.
- Fresh Edit seed behavior.

### 16.3 Input ordering

Required application contract:

| Request | Ordered image inputs |
|---|---|
| Generate, no references | None |
| Generate, one reference | Slot 1 |
| Generate, two references | Slot 1, slot 2 |
| Edit, no sketch | Current image |
| Edit, with sketch | Current image, white-background sketch |

The Edit UI may refer to the attachment as “the sketch.” Concise tooltip guidance may explain that the current image is image 1 and the sketch is image 2 if useful, without rewriting the user’s text.

### 16.4 Adapter validation

Verify that the repository’s pinned model/adapter supports the required two-image Edit request.

This specification defines the required contract; it does not assume that adding another image argument to the current code is sufficient.

If support is absent:

- Report the exact limitation.
- Resolve it through a focused adapter/dependency change.
- Do not silently composite the sketch onto the base.
- Do not fall back to text-only Edit.
- Do not add hidden prompt instructions as compensation.

### 16.5 Runtime invariants

Preserve:

- One serialized model invocation boundary.
- Existing model-family caching.
- Cancellation while queued or running.
- No published output after cancellation.
- No reuse of interrupted model state.
- Secure immutable source snapshots.

---

## 17. Implementation Responsibilities

### Domain

- Draft and sketch records.
- Reference unions.
- Provenance extensions.
- Validation.
- Historical provenance types within the new schema.

### Storage

- Immutable sketch persistence.
- Safe loading and snapshots.
- New-schema-only persistence and rejection of unsupported schemas.
- Save As.
- Reachability and cleanup.
- Asset/manifest transaction support.

### Application

- Authoritative version-local draft state.
- Draft commands and replacement Undo.
- Request capture.
- Paired consumption.
- History recall.
- Image Undo/Redo reconciliation.
- Removal of executable Refine workflow.

### Generation

- Ordered image-input contract.
- White export integration.
- Existing prompt policy.
- Model capability validation.

### UI

- Sketch dialog and tools.
- Reference slot affordances.
- Edit attachment.
- History thumbnails.
- Context-safe rendering.
- Evolve tab removal.

Widgets must not own independent persisted truth or perform model/storage operations directly.

### Final delivery operation

- Back up and convert Escape To Earth once, following Section 15.
- Keep conversion tooling outside maintained application code; no runtime migration responsibility belongs to Storage or any other production layer.

---

## 18. Acceptance Tests

Ordinary automated tests must use fakes and must not download or invoke models.

### 18.1 Drawing and export

Verify:

- Transparent pixels reveal the Edit base.
- Generate editor never displays the current image.
- White paint remains opaque.
- Eraser restores transparency.
- White export contains no base-image pixels.
- Thumbnails contain no editor overlays.
- Empty drawings cannot be attached.
- Output-tier changes do not alter drawing coordinates.
- Zoom does not change stored stroke width.
- Use Sketch commits once; Cancel commits nothing.
- Reopening preserves exact drawing pixels.
- Historical sketches are not mutated by editing recalls.

### 18.2 Reference slots

Verify:

- Every card/sketch combination up to two references.
- Correct model ordering.
- No duplicate image submission.
- Slot promotion and label updates.
- No Description rewriting.
- Existing card eligibility constraints.
- Generate success preserves sketch references.
- Replacing a reference is undoable.

### 18.3 Draft isolation

Verify:

- Different cards retain different text.
- Different versions retain different text.
- Different cards/versions retain different sketches.
- Returning restores the exact pair.
- Rerender does not create a new draft generation.
- Reopening the bundle restores committed drafts.
- Opening another stack cannot leak previous drafts.
- Duplicate card/version starts with an empty Edit draft.
- Base replacement preserves the draft without a warning.

### 18.4 History recall

Verify:

- Text-only recall removes the sketch.
- Sketch-bearing recall restores both inputs.
- Recall replacement is reversible as a pair.
- Identical-content recall still counts as newer input.
- Recall does not submit, navigate, or change Style/output size.
- Reopened recall displays the current base.
- Missing historical sketch leaves the current draft unchanged.

### 18.5 Completion and races

Verify:

- Durable success clears only the matching pair.
- Model success before save does not consume the pair.
- Failure/cancellation retains the draft.
- Delayed save retry consumes at most once.
- Newer drafts survive late callbacks.
- Another card’s draft is never cleared.
- Source replacement rejects stale work.
- Sketch replacement rejects/cancels stale work.
- Passive autosave/rerender does not cancel valid work.
- Temporary files are released safely.

### 18.6 Undo and version creation

Verify:

- Undo restores a consumed pair when eligible.
- Newer text or sketch blocks automatic restoration.
- Intentional newer empty drafts are protected.
- Redo clears only the exact automatically restored pair.
- Version creation does not resurrect consumed drafts.
- Undo version creation and Undo Edit remain separate boundaries.
- Newer unrelated drafts survive document Undo.

### 18.7 Storage and historical provenance

Verify:

- New-schema round trips and rejection of schema 13 and older without rewriting the bundle.
- Historical Refine load/save/display using new-schema fixtures.
- Editing an old Refine background.
- Duplicating old Refine provenance.
- All sketch roots participate in retention.
- Save As creates an independent bundle.
- Deletion/Undo restores assets.
- Cleanup rejects changed file identity.
- Symlink/path traversal rejection.
- Corrupt PNG and dimension-limit rejection.
- Rollback after asset/manifest failure.

Separately, as the final delivery step, verify the complete schema 13 backup and one-off Escape To Earth conversion against Section 15. This is not an automated runtime-migration test or a reason to retain an old-schema reader.

### 18.8 Evolve removal

Verify:

- No Evolve tab or executable action.
- No new Refine requests can be dispatched.
- Plain Generate still uses its supported model family.
- Edit and reference-backed Generate remain functional.
- Historical Refine facts are retained.
- No active prompt path reconstructs an image from accumulated Edit instructions.

---

## 19. Live Model Evaluation

Use the production adapter, prompt builders, snapshotting, and export code.

Do not fork behavior for evaluations.

### 19.1 Required cases

1. Generate from a single rough layout sketch.
2. Generate from a card reference plus a sketch.
3. Generate from two sketches.
4. Edit using positional marks over the current image.
5. Edit using a standalone object/design sketch.
6. Edit using a sketch with multiple colors and an explicit authored explanation.
7. Recall a sketch against a changed base.
8. Use opaque white paint and erasing in a drawing.
9. Run representative cases on each supported image-model configuration.

### 19.2 Evaluate

Assess:

- Whether intended additions or changes are recognizable.
- Whether rough marks are interpreted as guidance rather than literally copied when that is the authored intent.
- Whether the white export becomes an unwanted replacement background.
- Whether “the sketch” and positional image references are understood.
- Whether unrelated content remains reasonably consistent.
- Whether output size and aspect ratio remain correct.

Record:

- Exact prompt.
- Ordered input identities.
- Exported sketch.
- Output.
- Model/dependency settings.
- Human rubric results.

### 19.3 Release gate

There must be evidence that the required model-input path works without hidden sketch instructions.

If quality is inadequate, report the limitation and revisit the model/input approach explicitly. Do not conceal the limitation behind new warnings, color rules, or automatic prompt rewriting.

---

## 20. Suggested Delivery Sequence

### Phase 1 — Evolve removal and historical provenance

- Remove executable Evolve surfaces.
- Preserve historical Refine records.
- Update documentation and tests.

### Phase 2 — Version-local persisted Edit drafts

- Fix cross-card/version leakage.
- Add draft identity and persistence.
- Establish draft Undo and safe image-history reconciliation.
- Preserve existing text-only Edit behavior.

### Phase 3 — Sketch domain/storage/editor

- Add immutable sketch assets.
- Add RGBA editor and white export.
- Add transactional attachment changes and cleanup.

### Phase 4 — Generate sketch references

- Extend slots and provenance.
- Validate ordering and model contracts.

### Phase 5 — Edit sketch integration

- Add attachment UI.
- Extend request snapshots and accepted history.
- Implement paired consumption and recall.

### Phase 6 — Evaluation and hardening

- Live model evaluation.
- Failure injection.
- New-schema fixtures retaining historical provenance facts.
- Accessibility and lifecycle testing.
- Final documentation updates.

Each phase should preserve a usable application and focused review boundaries.

### Final step — Back up and convert Escape To Earth

- After all six phases and their reviews are complete, create and verify a complete schema 13 backup of the working stack.
- Perform the one-off conversion and preservation checks from Section 15.
- Leave the backup untouched and keep migration logic out of the application.

---

## 21. Original Full-Feature Definition of Done (Parked)

This section records the completion criteria for the parked sketch proposal; it
is not the completion criterion for the adopted reduced work.

- Generate and Edit are the only executable image-authoring operations.
- Historical Evolve images remain valid and usable.
- Generate supports any allowed combination of card and sketch references.
- Edit supports one optional sketch alongside its instruction.
- Editable sketches retain transparency.
- Model sketch inputs are flattened over white without the base.
- Drafts remain local to their card version and survive reopening.
- Base-image changes preserve sketches without warnings or restrictions.
- History recalls and consumption operate on the complete instruction/sketch pair.
- Undo/Redo and Create New Version protect newer drafts.
- Sketch assets remain available while reachable and are reclaimed safely afterward.
- Request ordering and model support are verified.
- Automated tests pass and representative live evaluations are recorded.
- `README.md` and durable repository guidance reflect the new behavior.
- Production loading supports only the new schema, without legacy-schema readers or runtime migration.
- Escape To Earth has been converted as the final step, with its content and assets preserved and an untouched, verified schema 13 backup retained.

**Final product rule: an Edit draft belongs to a card version, consists of an instruction and an optional sketch, and is applied to that version’s current image. The user—not a warning system—decides whether the sketch is appropriate.**
