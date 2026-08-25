"""Offline JSON, CSV, image, and HTML evaluation report rendering."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

from hypergen.evaluation.manifest import safe_run_path
from hypergen.generation.errors import GenerationError

REPORT_VERSION = "eval-report-v1"


class ReportRenderingError(GenerationError):
    """Offline report rendering failed after suite data was retained."""

    def __init__(self, result_path: Path, cause: BaseException) -> None:
        self.result_path = result_path
        self.cause = cause
        super().__init__(f"could not render reports for {result_path}: {cause}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def annotate_hotspots(
    source_path: Path,
    output_path: Path,
    proposals: Sequence[Mapping[str, object]],
) -> Path:
    """Draw normalized production hotspot polygons and labels."""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de")
    for index, proposal in enumerate(proposals):
        color = colors[index % len(colors)]
        label = str(proposal.get("label", f"Hotspot {index + 1}"))
        for polygon in proposal.get("polygons", []):  # type: ignore[union-attr]
            points = [
                (
                    round(float(point["x"]) * image.width),
                    round(float(point["y"]) * image.height),
                )
                for point in polygon["points"]
            ]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=color, width=max(2, image.width // 300))
                draw.text((points[0][0] + 3, points[0][1] + 3), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def create_contact_sheet(
    entries: Sequence[tuple[Path, str]],
    output_path: Path,
    *,
    cell_size: tuple[int, int] = (360, 280),
    columns: int = 3,
) -> Path | None:
    """Create a labeled Pillow contact sheet, or no file for no entries."""
    if not entries:
        return None
    rows = (len(entries) + columns - 1) // columns
    label_height = 36
    sheet = Image.new(
        "RGB",
        (cell_size[0] * columns, (cell_size[1] + label_height) * rows),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(entries):
        with Image.open(path) as source:
            thumb = ImageOps.contain(source.convert("RGB"), cell_size)
        x = (index % columns) * cell_size[0]
        y = (index // columns) * (cell_size[1] + label_height)
        image_x = x + (cell_size[0] - thumb.width) // 2
        image_y = y + (cell_size[1] - thumb.height) // 2
        sheet.paste(thumb, (image_x, image_y))
        draw.text((x + 8, y + cell_size[1] + 8), label[:55], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


def _phase_row(
    *,
    suite: str,
    axis: str,
    case_id: object = "",
    ollama_model: object = "",
    mflux_model: object = "",
    phase: str = "",
    record: Mapping[str, object],
) -> dict[str, object]:
    metrics = record.get("metrics") or record.get("cold_metrics") or {}
    token_usage = record.get("token_usage") or {}
    failure = record.get("failure") or {}
    rubric = record.get("human_rubric") or record.get("rubric")
    elapsed = record.get("elapsed_seconds")
    if elapsed is None and isinstance(metrics, Mapping):
        elapsed = metrics.get("elapsed_seconds")
    if elapsed is None:
        elapsed = record.get("total_duration_seconds") or record.get("inference_duration_seconds")
    nested_result = record.get("result")
    if elapsed is None and isinstance(nested_result, Mapping):
        elapsed = nested_result.get("duration_seconds")
    prompt_tokens = token_usage.get("prompt_tokens") if isinstance(token_usage, Mapping) else None
    output_tokens = token_usage.get("output_tokens") if isinstance(token_usage, Mapping) else None
    prompt_tokens = prompt_tokens if prompt_tokens is not None else record.get("prompt_eval_count")
    output_tokens = output_tokens if output_tokens is not None else record.get("eval_count")
    timings = {key: value for key, value in record.items() if "duration" in key or key == "timings"}
    if isinstance(nested_result, Mapping):
        timings.update({key: value for key, value in nested_result.items() if "duration" in key})
    render_prompt = record.get("render_prompt") or record.get("prompt")
    interactive_subjects = record.get("interactive_subjects")
    if isinstance(nested_result, Mapping):
        render_prompt = (
            render_prompt or nested_result.get("render_prompt") or nested_result.get("prompt")
        )
        interactive_subjects = interactive_subjects or nested_result.get("interactive_subjects")
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        render_prompt = render_prompt or metadata.get("render_prompt")
    return {
        "suite": suite,
        "axis": axis,
        "case_id": case_id,
        "ollama_model": ollama_model,
        "mflux_model": mflux_model,
        "phase": phase,
        "status": record.get("status", "success"),
        "failure_classification": (
            failure.get("classification", "") if isinstance(failure, Mapping) else ""
        ),
        "elapsed_seconds": elapsed,
        "timings": timings,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "render_prompt": render_prompt,
        "interactive_subjects": interactive_subjects,
        "structured_valid": record.get("structured_valid"),
        "repetition": record.get("repetition"),
        "schema_limits": record.get("schema_limits"),
        "geometry": record.get("geometry"),
        "destinations": record.get("destinations"),
        "warnings": record.get("warnings", []),
        "human_rubric": rubric,
    }


def _summary_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    version = str(result.get("result_version", ""))
    rows: list[dict[str, object]] = []
    if version.startswith("image-result"):
        for item in result.get("prompt_axis", []):  # type: ignore[union-attr]
            for phase in ("cold", "warm"):
                record = dict(item[phase])
                record["metrics"] = item[f"{phase}_metrics"]
                record["rubric"] = item.get("rubric")
                rows.append(
                    _phase_row(
                        suite="images",
                        axis="ollama_prompt",
                        case_id=item["case_id"],
                        ollama_model=item["model"],
                        phase=phase,
                        record=record,
                    )
                )
        for item in result.get("prompt_downstream_axis", []):  # type: ignore[union-attr]
            rows.append(
                _phase_row(
                    suite="images",
                    axis="prompt_downstream",
                    case_id=item["case_id"],
                    ollama_model=item["ollama_model"],
                    mflux_model=item["mflux_model"],
                    phase="generation",
                    record={**item["generation"], "rubric": item.get("rubric")},
                )
            )
        for item in result.get("mflux_axis", []):  # type: ignore[union-attr]
            for phase in ("cold", "warm"):
                rows.append(
                    _phase_row(
                        suite="images",
                        axis="mflux",
                        case_id=item["case_id"],
                        mflux_model=item["model"],
                        phase=phase,
                        record={
                            **item[phase],
                            "render_prompt": item.get("render_prompt") or item.get("fixed_prompt"),
                            "rubric": item.get("rubric"),
                        },
                    )
                )
    elif version.startswith("hotspot-result"):
        for item in result.get("models", []):  # type: ignore[union-attr]
            for phase in ("cold", "warm"):
                rows.append(
                    _phase_row(
                        suite="hotspots",
                        axis="ollama_hotspot",
                        case_id=item["case_id"],
                        ollama_model=item["model"],
                        phase=phase,
                        record=item[phase],
                    )
                )
        for item in result.get("ablations", []):  # type: ignore[union-attr]
            rows.append(
                _phase_row(
                    suite="hotspots",
                    axis=f"ablation:{item['name']}",
                    ollama_model=item["model"],
                    phase=str(item["name"]),
                    record=item["result"],
                )
            )
        for item in result.get("recorded_regressions", []):  # type: ignore[union-attr]
            rows.append(
                _phase_row(
                    suite="hotspots",
                    axis="recorded_regression",
                    phase=str(item["name"]),
                    record=item,
                )
            )
    elif version.startswith("e2e-result"):
        for item in result.get("candidates", []):  # type: ignore[union-attr]
            for stage in item.get("stages", []):
                rows.append(
                    _phase_row(
                        suite="e2e",
                        axis=str(stage["name"]),
                        case_id=item["case_id"],
                        ollama_model=item["ollama_model"],
                        mflux_model=item["mflux_model"],
                        phase=str(stage["name"]),
                        record={
                            **stage,
                            "render_prompt": item.get("render_prompt"),
                        },
                    )
                )
    elif version.startswith("image-prompt-benchmark-result"):
        for item in result.get("results", []):  # type: ignore[union-attr]
            rows.append(
                _phase_row(
                    suite="image_prompts",
                    axis="image_prompt_preparation",
                    case_id=item["case_id"],
                    ollama_model=item["model"],
                    phase=f"repetition-{item['repetition']}",
                    record={
                        **item,
                        "render_prompt": item.get("image_prompt"),
                        "rubric": item.get("rubric"),
                    },
                )
            )
    elif version.startswith("smoke-result"):
        for stage_name, stage in result.get("stages", {}).items():  # type: ignore[union-attr]
            if isinstance(stage, Mapping) and ("cold" in stage or "warm" in stage):
                for phase in ("cold", "warm"):
                    if phase in stage:
                        rows.append(
                            _phase_row(
                                suite="smoke",
                                axis=str(stage_name),
                                ollama_model=result.get("ollama_model", ""),
                                mflux_model=result.get("mflux_model", ""),
                                phase=phase,
                                record=stage[phase],
                            )
                        )
    failure = result.get("failure")
    if isinstance(failure, Mapping):
        rows.append(
            _phase_row(
                suite=str(result.get("suite", "evaluation")),
                axis=str(failure.get("stage", "run")),
                phase="failure",
                record={"status": "failed", "failure": failure},
            )
        )
    return rows


def _artifact_entries(
    run_dir: Path,
    result: Mapping[str, object],
) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    version = str(result.get("result_version", ""))
    if version.startswith("image-result"):
        for item in result.get("prompt_downstream_axis", []):  # type: ignore[union-attr]
            path_value = item["generation"].get("artifact_path")
            if path_value:
                path = safe_run_path(run_dir, path_value)
                entries.append((path, f"{item['case_id']}: {item['ollama_model']}"))
        for item in result.get("mflux_axis", []):  # type: ignore[union-attr]
            for phase in ("cold", "warm"):
                path_value = item[phase].get("artifact_path")
                if path_value:
                    path = safe_run_path(run_dir, path_value)
                    entries.append((path, f"{item['case_id']}: {item['model']} {phase}"))
    elif version.startswith("e2e-result"):
        for item in result.get("candidates", []):  # type: ignore[union-attr]
            path_value = item.get("artifact_path")
            if path_value:
                entries.append(
                    (
                        safe_run_path(run_dir, path_value),
                        f"{item['case_id']}: {item['ollama_model']}",
                    )
                )
    return entries


def _render_annotations(
    run_dir: Path,
    result: Mapping[str, object],
) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    version = str(result.get("result_version", ""))
    if version.startswith("hotspot-result"):
        cases = {item["case_id"]: item for item in result.get("cases", [])}  # type: ignore[union-attr]
        for item in result.get("models", []):  # type: ignore[union-attr]
            source = safe_run_path(run_dir, cases[item["case_id"]]["artifact_image_path"])
            for phase in ("cold", "warm"):
                record = item[phase]
                if record.get("status") != "success":
                    continue
                relative = (
                    Path("annotations")
                    / str(item["case_id"])
                    / f"{str(item['model']).replace(':', '-')}-{phase}.png"
                )
                output = safe_run_path(run_dir, relative)
                annotate_hotspots(source, output, record["result"]["proposals"])
                entries.append((output, f"{item['case_id']}: {item['model']} {phase}"))
    elif version.startswith("e2e-result"):
        for item in result.get("candidates", []):  # type: ignore[union-attr]
            if not item.get("artifact_path") or not item.get("hotspot_proposals"):
                continue
            source = safe_run_path(run_dir, item["artifact_path"])
            relative = (
                Path("annotations")
                / str(item["case_id"])
                / f"{str(item['ollama_model']).replace(':', '-')}.png"
            )
            output = safe_run_path(run_dir, relative)
            annotate_hotspots(source, output, item["hotspot_proposals"])
            entries.append((output, f"{item['case_id']}: {item['ollama_model']}"))
    return entries


def _format_cell(value: object) -> str:
    if value is None:
        return '<span class="unscored">UNSCORED / null</span>'
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value))


def _html_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "<p>No completed result rows.</p>"
    columns = tuple(rows[0])
    headers = "".join(
        f"<th>{html.escape(column.replace('_', ' ').title())}</th>" for column in columns
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_format_cell(row.get(column))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def _gallery(run_dir: Path, entries: Iterable[tuple[Path, str]]) -> str:
    figures = []
    for path, label in entries:
        relative = path.relative_to(run_dir).as_posix()
        figures.append(
            "<figure>"
            f'<img src="{html.escape(quote(relative))}" alt="{html.escape(label)}">'
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
        )
    return "".join(figures)


def _visual_entries(
    artifacts: Sequence[tuple[Path, str]],
    annotations: Sequence[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    """Prefer annotations while retaining artifacts without one."""
    annotations_by_label = {label: path for path, label in annotations}
    artifact_labels = {label for _, label in artifacts}
    entries = [(annotations_by_label.get(label, path), label) for path, label in artifacts]
    entries.extend((path, label) for path, label in annotations if label not in artifact_labels)
    return entries


def render_reports(result_path: Path) -> dict[str, Path]:
    """Render all reports solely from one suite result JSON file."""
    result_path = result_path.resolve()
    run_dir = result_path.parent
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = _summary_rows(result)
    annotations = _render_annotations(run_dir, result)
    artifacts = _artifact_entries(run_dir, result)
    sheet_entries = _visual_entries(artifacts, annotations)
    contact_sheet_path = run_dir / "contact-sheet.png"
    contact_sheet = create_contact_sheet(sheet_entries, contact_sheet_path)
    stage_metrics: list[dict[str, object]] = []
    for axis in sorted({str(row["axis"]) for row in rows}):
        axis_rows = [row for row in rows if row["axis"] == axis]
        failures = [row for row in axis_rows if row["status"] != "success"]
        stage_metrics.append(
            {
                "stage": axis,
                "count": len(axis_rows),
                "failure_rate": len(failures) / len(axis_rows),
                "timeout_rate": (
                    sum(row["failure_classification"] == "timeout" for row in axis_rows)
                    / len(axis_rows)
                ),
            }
        )
    summary = {
        "report_version": REPORT_VERSION,
        "result_version": result.get("result_version"),
        "suite_status": result.get("status", "completed"),
        "human_quality": {
            "primary_assessment": True,
            "note": "Null values are explicitly unscored and require human review.",
        },
        "stage_metrics": stage_metrics,
        "rows": rows,
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    csv_path = run_dir / "summary.csv"
    columns = (
        tuple(rows[0])
        if rows
        else (
            "suite",
            "axis",
            "case_id",
            "ollama_model",
            "mflux_model",
            "phase",
            "status",
            "failure_classification",
            "elapsed_seconds",
            "timings",
            "prompt_tokens",
            "output_tokens",
            "warnings",
            "human_rubric",
        )
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    title = f"HyperGen {result.get('suite', result.get('result_version', 'evaluation'))} report"
    contact_link = (
        '<p><a href="contact-sheet.png">Open contact sheet</a></p>' if contact_sheet else ""
    )
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#18202a}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3dc;padding:.45rem;vertical-align:top}} th{{background:#edf2f7}}
.rubric{{border:3px solid #8b5cf6;background:#f5f3ff;padding:1rem;margin:1rem 0}}
.unscored{{color:#9a3412;font-weight:700}} .gallery{{display:flex;flex-wrap:wrap;gap:1rem}}
figure{{margin:0;max-width:360px}} img{{max-width:100%;height:auto}} code{{white-space:pre-wrap}}
</style></head><body><h1>{html.escape(title)}</h1>
<div class="rubric"><h2>Human quality rubric is primary</h2>
<p>Every <strong>UNSCORED / null</strong> value requires manual assessment. Automated validity,
geometry, timings, and token counts do not substitute for image or hotspot quality review.</p></div>
<p>Status: <strong>{html.escape(str(summary["suite_status"]))}</strong></p>
{contact_link}<h2>Structured results</h2>{_html_table(rows)}
<h2>Visual artifacts</h2><div class="gallery">{_gallery(run_dir, sheet_entries)}</div>
</body></html>"""
    html_path = run_dir / "report.html"
    html_path.write_text(report_html, encoding="utf-8")
    paths = {"json": summary_path, "csv": csv_path, "html": html_path}
    if contact_sheet:
        paths["contact_sheet"] = contact_sheet
    return paths


def render_reports_checked(result_path: Path) -> dict[str, Path]:
    """Render reports and expose a stable evaluation-layer failure."""
    try:
        return render_reports(result_path)
    except Exception as error:
        raise ReportRenderingError(result_path, error) from error
