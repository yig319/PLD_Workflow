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
        json.dump(normalized, handle, indent=2)

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


def _instrument_spec() -> List[Tuple[str, List[Tuple[str, Tuple[str, ...]]]]]:
    """Return section definitions for the Instrument card (shared across targets)."""
    return [
        (
            "Target Setup",
            [
                ("Offset X (mm)", ("Offset X (mm)", "Offset X")),
                ("Offset Y (mm)", ("Offset Y (mm)", "Offset Y")),
                ("Scan Diameter (mm)", ("Scan Diameter (mm)", "Scan Diameter")),
                ("Scan Speed X/Z (mm/s)", ("Scan Speed X/Z (mm/s)",)),
            ],
        ),
        (
            "Heater Position",
            [
                ("Position X (mm)", ("Heater Position X (mm)", "Heater Position X")),
                ("Position Y (mm)", ("Heater Position Y (mm)", "Heater Position Y")),
                ("Position Z (mm)", ("Heater Position Z (mm)", "Heater Position Z")),
            ],
        ),
        (
            "Mask and Spot",
            [
                ("Mask Width (mm)", ("Mask Width (mm)", "Mask Width")),
                ("Mask Height (mm)", ("Mask Height (mm)", "Mask Height")),
                ("Mask Area (mm^2)", ("Mask Area (mm^2)", "Mask Area")),
                ("Spot Width (mm)", ("Spot Width (mm)", "Spot Width")),
                ("Spot Height (mm)", ("Spot Height (mm)", "Spot Height")),
                ("Spot Area (mm^2)", ("Spot Area (mm^2)", "Spot Area")),
                ("Magnification (x)", ("Magnification (x)", "Magnification")),
            ],
        ),
        (
            "Laser",
            [
                ("Laser Voltage (kV)", ("Laser Voltage (kV)", "Laser Voltage(kV)")),
                ("Laser Energy (mJ)", ("Laser Energy (mJ)", "Laser Energy(mJ)")),
                (
                    "Measured Energy (mJ)",
                    (
                        "Measured Energy (mJ)",
                        "Measured Energy(mJ)",
                        "Targeted Measured Energy (mJ)",
                        "Targeted Measured Energy(mJ)",
                        "Measured Energy Mean(mJ)",
                    ),
                ),
                ("Fluence (J/cm^2)", ("Fluence (J/cm^2)", "Fluence")),
            ],
        ),
        (
            "RHEED Adjustment",
            [
                ("Tilt (deg)", ("Tilt (deg)", "Tilt", "Tile")),
                ("Azimuth (deg)", ("Azimuth (deg)", "Azimuth")),
            ],
        ),
    ]


def _preparation_spec() -> List[Tuple[str, List[Tuple[str, Tuple[str, ...]]]]]:
    """Return section definitions for the Preparation card (shared across targets)."""
    return [
        (
            "Pre-Annealing",
            [
                (
                    "Heat Rate (\N{DEGREE SIGN}C/min)",
                    (
                        "Pre-Annealing Heat Rate (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing-Heating-Speed(\N{DEGREE SIGN}C/min)",
                    ),
                ),
                (
                    "Anneal Temperature (\N{DEGREE SIGN}C)",
                    (
                        "Pre-Annealing Temperature (\N{DEGREE SIGN}C)",
                        "Pre-Annealing-Temperature(\N{DEGREE SIGN}C)",
                        "Pre-Annealing Temperature (degC)",
                    ),
                ),
                (
                    "Hold Time (min)",
                    (
                        "Pre-Annealing Hold Time (min)",
                        "Pre-Annealing Time (min)",
                        "Pre-Annealing-Time(min)",
                        "Pre-Annealing-Time",
                    ),
                ),
                (
                    "Pressure (mbar)",
                    ("Pre-Annealing Pressure (mbar)",),
                ),
                (
                    "Pressure (mTorr)",
                    (
                        "Pre-Annealing Pressure (mTorr)",
                        "Pre-Annealing Atmosphere Pressure (mTorr)",
                        "Pre-Annealing-Atmosphere-Pressure(mTorr)",
                    ),
                ),
            ],
        ),
        (
            "Growth Condition",
            [
                (
                    "Heating Rate (\N{DEGREE SIGN}C/min)",
                    (
                        "Growth Condition Heating Rate (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing Heating Rate (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing Cool/Heat Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                        "Pre-Annealing Cooling Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                    ),
                ),
                (
                    "Temperature (\N{DEGREE SIGN}C)",
                    (
                        "Ablation Temperature (\N{DEGREE SIGN}C)",
                        "Ablation-Temperature(\N{DEGREE SIGN}C)",
                        "Ablation Temperature (degC)",
                        "Pre-Annealing Growth Temperature (\N{DEGREE SIGN}C)",
                        "Pre-Annealing Growth Temperature (degC)",
                        "Growth Temperature (\N{DEGREE SIGN}C)",
                        "Growth Temperature (degC)",
                    ),
                ),
                (
                    "Atmosphere Gas",
                    ("Ablation Atmosphere Gas", "Ablation-Atmosphere Gas"),
                ),
                (
                    "Pressure (mbar)",
                    ("Growth Pressure (mbar)", "Ablation Pressure (mbar)"),
                ),
                (
                    "Pressure (mTorr)",
                    ("Growth Pressure (mTorr)", "Ablation Pressure (mTorr)", "Ablation-Pressure(mTorr)"),
                ),
            ],
        ),
        (
            "Cool Down",
            [
                (
                    "Cooling Rate (\N{DEGREE SIGN}C/min)",
                    (
                        "Post-Annealing Cooling Rate (\N{DEGREE SIGN}C/min)",
                        "Post-Annealing Cool Rate (\N{DEGREE SIGN}C/min)",
                    ),
                ),
                (
                    "Pressure (mbar)",
                    ("Post-Annealing Pressure (mbar)",),
                ),
                (
                    "Pressure (mTorr)",
                    (
                        "Post-Annealing Pressure (mTorr)",
                        "Post-Annealing Atmosphere Pressure (mTorr)",
                        "Post-Annealing-Atmosphere-Pressure(mTorr)",
                    ),
                ),
            ],
        ),
    ]


def _deposition_spec() -> List[Tuple[str, List[Tuple[str, Tuple[str, ...]]]]]:
    """Return section definitions for per-target Deposition cards."""
    return [
        (
            "Pre-Ablation",
            [
                (
                    "Frequency (Hz)",
                    ("Pre-Ablation Frequency (Hz)", "Pre-Ablation-Frequency(Hz)"),
                ),
                (
                    "Pulses (count)",
                    ("Pre-Ablation Pulses (count)", "Pre-Ablation-Pulses"),
                ),
            ],
        ),
        (
            "Ablation",
            [
                ("Frequency (Hz)", ("Ablation Frequency (Hz)", "Ablation-Frequency(Hz)")),
                ("Pulses (count)", ("Ablation Pulses (count)", "Ablation-Pulses")),
            ],
        ),
    ]


def _first_present_key(data: Dict[str, Any], keys: Tuple[str, ...]) -> str | None:
    """Return the first matching key present in one target dictionary."""
    for key in keys:
        if key in data:
            return key
    return None


def _render_sections(
    target_data: Dict[str, Any],
    spec: List[Tuple[str, List[Tuple[str, Tuple[str, ...]]]]],
) -> str:
    """Render grouped sub-section tables from target data using the given spec."""
    sections: List[str] = []

    for section_title, section_rows in spec:
        rows: List[List[Any]] = []
        for display_name, candidate_keys in section_rows:
            present_key = _first_present_key(target_data, candidate_keys)
            if present_key is None:
                continue
            rows.append([display_name, target_data[present_key]])
        if not rows:
            continue
        table_markup = _render_html_table(["parameter", "value"], rows)
        sections.append(
            "<section class='subcard'>"
            f"<h3>{html.escape(section_title)}</h3>"
            f"{table_markup}"
            "</section>"
        )

    return "".join(sections)


def write_html_report(file_path: str, data: Any) -> None:
    """Write form data to a styled HTML report matching the form UI layout."""
    cards: List[str] = []

    if not isinstance(data, dict):
        headers, rows = _to_table_rows(data)
        cards.append(
            "<section class='card'>"
            "<h2>Parameters</h2>"
            f"{_render_html_table(headers, rows)}"
            "</section>"
        )
        _write_html_document(file_path, cards)
        return

    header_data = data.get("header")

    # 1. Header card
    if isinstance(header_data, dict):
        headers, rows = _to_table_rows(header_data)
        cards.append(
            "<section class='card'>"
            "<h2>Header</h2>"
            f"{_render_html_table(headers, rows)}"
            "</section>"
        )

    # Sort target keys
    target_keys = [key for key in data.keys() if key.lower().startswith("target_")]
    target_keys.sort(key=lambda key: int(key.split("_")[1]) if key.split("_")[1].isdigit() else key)

    if target_keys:
        first_target = data[target_keys[0]]

        # 2. Instrument card (shared — read from first target)
        if isinstance(first_target, dict):
            instrument_html = _render_sections(first_target, _instrument_spec())
            if instrument_html:
                cards.append(
                    "<section class='card'>"
                    "<h2>Instrument</h2>"
                    f"<div class='section-grid'>{instrument_html}</div>"
                    "</section>"
                )

        # 3. Preparation card (shared — read from first target)
        if isinstance(first_target, dict):
            prep_html = _render_sections(first_target, _preparation_spec())
            if prep_html:
                cards.append(
                    "<section class='card'>"
                    "<h2>Preparation</h2>"
                    f"<div class='section-grid'>{prep_html}</div>"
                    "</section>"
                )

        # 4. Deposition cards (one per target)
        for target_key in target_keys:
            target_data = data.get(target_key)
            if not isinstance(target_data, dict):
                continue
            # Show target material in the card title when available
            material = target_data.get("Target Material", "")
            title = f"Deposition - {material}" if material else f"Deposition - {target_key.replace('_', ' ').title()}"
            depo_html = _render_sections(target_data, _deposition_spec())
            if depo_html:
                cards.append(
                    "<section class='card'>"
                    f"<h2>{html.escape(title)}</h2>"
                    f"<div class='section-grid'>{depo_html}</div>"
                    "</section>"
                )

    # 5. Notes card
    if isinstance(header_data, dict):
        notes = str(header_data.get("Notes", "")).strip()
        if notes:
            cards.append(
                "<section class='card'>"
                "<h2>Notes</h2>"
                f"{_render_html_table(['Notes'], [[notes]])}"
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
      --bg: #edf3f8;
      --bg-soft: #f9fbfd;
      --card: rgba(255, 255, 255, 0.96);
      --line: #d6dfe9;
      --text: #182230;
      --muted: #526178;
      --head: #eaf1f7;
      --accent: #295979;
      --accent-soft: #f4f8fb;
      --shadow: 0 14px 32px rgba(24, 34, 48, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, var(--bg) 0%, var(--bg-soft) 100%);
      color: var(--text);
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
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
      border-radius: 14px;
      padding: 14px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
      color: var(--accent);
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .subcard {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: var(--accent-soft);
    }}
    .subcard h3 {{
      margin: 0 0 8px 0;
      font-size: 14px;
      color: var(--accent);
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
