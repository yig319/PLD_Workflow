"""Utilities for saving PLD parameter data to JSON and HTML."""

from __future__ import annotations

import datetime
import html
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class SaveParametersResult:
    """Result object for save operations."""

    json_path: str
    html_path: str | None
    html_error: str | None = None


def build_default_file_stem(growth_id: str, user_name: str, date_text: str) -> str:
    """Build output file stem from core header fields."""
    growth_id_clean = growth_id.strip()
    user_name_clean = user_name.strip()
    date_stamp = "".join(date_text.split("/")).strip()

    if growth_id_clean or user_name_clean:
        return f"{growth_id_clean}_{user_name_clean}_{date_stamp}".strip("_")

    if not date_stamp:
        date_stamp = datetime.datetime.today().strftime("%m%d%Y")
    return f"growth_record_{date_stamp}"


def coerce_numeric_values(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert numeric-looking string values to floats in a nested section dictionary."""
    converted: Dict[str, Dict[str, Any]] = {}
    for section_name, section in data.items():
        converted[section_name] = {}
        for key, value in section.items():
            converted[section_name][key] = _coerce_float(value)
    return converted


def save_parameters_json_and_html(
    info_dict: Dict[str, Dict[str, Any]],
    output_dir: str,
    file_stem: str,
) -> SaveParametersResult:
    """Save parameters into JSON and HTML in the same directory.

    JSON write errors are raised to the caller. HTML write errors are captured in the
    returned result so callers can report partial success (JSON-only).
    """
    os.makedirs(output_dir, exist_ok=True)

    normalized = coerce_numeric_values(info_dict)
    json_path = os.path.join(output_dir, f"{file_stem}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle)

    html_path = os.path.join(output_dir, f"{file_stem}.html")
    try:
        write_html_report(html_path, normalized)
        return SaveParametersResult(json_path=json_path, html_path=html_path)
    except Exception as exc:  # noqa: BLE001
        return SaveParametersResult(json_path=json_path, html_path=None, html_error=str(exc))


def _coerce_float(value: Any) -> Any:
    """Return a float when text is numeric; otherwise return original value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _has_nested_container(value: Any) -> bool:
    """Return True when dict/list contains nested dict/list values."""
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, (dict, list)) or _has_nested_container(item):
                return True
        return False
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)) or _has_nested_container(item):
                return True
        return False
    return False


def _flatten_rows(value: Any, prefix: str = "") -> List[Dict[str, Any]]:
    """Flatten nested dict/list data into parameter/value rows."""
    rows: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_rows(item, key_path))
        return rows

    if isinstance(value, list):
        if not value:
            rows.append({"parameter": prefix, "value": ""})
            return rows
        for index, item in enumerate(value):
            key_path = f"{prefix}[{index}]"
            rows.extend(_flatten_rows(item, key_path))
        return rows

    rows.append({"parameter": prefix, "value": value})
    return rows


def _to_table_rows(data: Any) -> Tuple[List[str], List[List[Any]]]:
    """Convert input data into simple headers/rows for table rendering."""
    if isinstance(data, dict):
        if _has_nested_container(data):
            flat = _flatten_rows(data)
            return ["parameter", "value"], [[r["parameter"], r["value"]] for r in flat]
        return ["parameter", "value"], [[k, v] for k, v in data.items()]

    if isinstance(data, list):
        if data and all(isinstance(item, dict) and not _has_nested_container(item) for item in data):
            headers: List[str] = []
            for item in data:
                for key in item.keys():
                    if key not in headers:
                        headers.append(key)
            rows = [[item.get(h, "") for h in headers] for item in data]
            return headers, rows

        if any(isinstance(item, (dict, list)) for item in data):
            flat = _flatten_rows(data)
            return ["parameter", "value"], [[r["parameter"], r["value"]] for r in flat]

        return ["value"], [[item] for item in data]

    return ["value"], [[data]]


def _render_html_table(headers: List[str], rows: List[List[Any]], show_header: bool = False) -> str:
    """Build HTML markup for one data table."""
    header_cells = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    row_markup: List[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
        row_markup.append(f"<tr>{cells}</tr>")
    colspan = max(1, len(headers))
    body = "".join(row_markup) if row_markup else f"<tr><td colspan='{colspan}'>&nbsp;</td></tr>"
    if show_header:
        return f"<table><thead><tr>{header_cells}</tr></thead><tbody>{body}</tbody></table>"
    return f"<table><tbody>{body}</tbody></table>"


def _target_section_spec() -> List[Tuple[str, List[str]]]:
    """Return ordered target section definitions used in HTML export."""
    return [
        (
            "Target Parameters",
            [
                "Target Material",
                "Offset X (mm)",
                "Offset Y (mm)",
                "Scan Diameter (mm)",
                "Scan Speed X/Z (mm/s)",
            ],
        ),
        (
            "Heater Parameter",
            [
                "Heater Position X (mm)",
                "Heater Position Y (mm)",
                "Heater Position Z (mm)",
                "Tilt (deg)",
                "Azimuth (deg)",
            ],
        ),
        (
            "Laser and Mask",
            [
                "Laser Voltage (kV)",
                "Laser Energy (mJ)",
                "Targeted Measured Energy (mJ)",
                "Fluence (J/cm^2)",
            ],
        ),
        (
            "Mask and Spot",
            [
                "Mask Width (mm)",
                "Mask Height (mm)",
                "Mask Area (mm^2)",
                "Spot Width (mm)",
                "Spot Height (mm)",
                "Spot Area (mm^2)",
                "Magnification (x)",
            ],
        ),
        (
            "Pre-annealing",
            [
                "Pre-Annealing Temperature (\N{DEGREE SIGN}C)",
                "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)",
                "Pre-Annealing Time (min)",
                "Pre-Annealing Atmosphere Pressure (mTorr)",
            ],
        ),
        (
            "Ablation",
            [
                "Pre-Ablation Pulses (count)",
                "Ablation Temperature (\N{DEGREE SIGN}C)",
                "Ablation Pressure (mTorr)",
                "Ablation Atmosphere Gas",
                "Ablation Frequency (Hz)",
                "Ablation Pulses (count)",
            ],
        ),
    ]


def _render_target_sections(target_data: Dict[str, Any]) -> str:
    """Render grouped target tables by form section for HTML output."""
    sections: List[str] = []
    used_keys: set[str] = set()

    for section_title, section_keys in _target_section_spec():
        rows = [[key, target_data[key]] for key in section_keys if key in target_data]
        if not rows:
            continue
        used_keys.update(key for key in section_keys if key in target_data)
        table_markup = _render_html_table(["parameter", "value"], rows)
        sections.append(
            "<section class='subcard'>"
            f"<h3>{html.escape(section_title)}</h3>"
            f"{table_markup}"
            "</section>"
        )

    remaining_rows = [[key, value] for key, value in target_data.items() if key not in used_keys]
    if remaining_rows:
        table_markup = _render_html_table(["parameter", "value"], remaining_rows)
        sections.append(
            "<section class='subcard'>"
            "<h3>Other</h3>"
            f"{table_markup}"
            "</section>"
        )

    return "".join(sections)


def write_html_report(file_path: str, data: Any) -> None:
    """Write form data to a styled HTML report for OneNote import."""
    cards: List[str] = []
    if isinstance(data, dict):
        header_data = data.get("header")
        if isinstance(header_data, dict):
            headers, rows = _to_table_rows(header_data)
            cards.append(
                "<section class='card'>"
                "<h2>Header</h2>"
                f"{_render_html_table(headers, rows)}"
                "</section>"
            )

        target_keys = [key for key in data.keys() if key.lower().startswith("target_")]
        target_keys.sort(key=lambda key: int(key.split("_")[1]) if key.split("_")[1].isdigit() else key)
        for target_key in target_keys:
            target_data = data.get(target_key)
            if not isinstance(target_data, dict):
                headers, rows = _to_table_rows(target_data)
                target_markup = _render_html_table(headers, rows)
            else:
                target_markup = f"<div class='section-grid'>{_render_target_sections(target_data)}</div>"

            cards.append(
                "<section class='card'>"
                f"<h2>{html.escape(target_key.replace('_', ' ').title())}</h2>"
                f"{target_markup}"
                "</section>"
            )

        other_sections = [
            (key, value)
            for key, value in data.items()
            if key != "header" and key not in target_keys
        ]
        for section_name, section_data in other_sections:
            headers, rows = _to_table_rows(section_data)
            cards.append(
                "<section class='card'>"
                f"<h2>{html.escape(str(section_name))}</h2>"
                f"{_render_html_table(headers, rows)}"
                "</section>"
            )
    else:
        headers, rows = _to_table_rows(data)
        cards.append(
            "<section class='card'>"
            "<h2>Parameters</h2>"
            f"{_render_html_table(headers, rows)}"
            "</section>"
        )

    page_title = "PLD Growth Parameters"
    now_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --line: #d8e1ec;
      --text: #182230;
      --muted: #526178;
      --head: #ebf2f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    .meta {{
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
      color: #243955;
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .subcard {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfdff;
    }}
    .subcard h3 {{
      margin: 0 0 8px 0;
      font-size: 14px;
      color: #2d4b72;
      letter-spacing: 0.2px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: var(--head);
      text-transform: capitalize;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(page_title)}</h1>
  <div class="meta">Exported: {html.escape(now_stamp)}</div>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(document)
