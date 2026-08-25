"""Pulse-history calculations for PLD growth records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


MAX_PULSE_COUNT = 2_000_000_000


@dataclass(frozen=True)
class PulseHistoryResult:
    """Resolved target-pulse state immediately before a new run."""

    before_pulses: int
    matching_records: int
    source_path: Path | None = None
    source_growth_id: str = ""
    source_timestamp: datetime | None = None
    used_material_fallback: bool = False
    warning: str = ""


@dataclass(frozen=True)
class _PulseRecord:
    timestamp: datetime
    path: Path
    growth_id: str
    target_id: str
    material: str
    before_pulses: int | None
    after_pulses: int | None
    on_target_this_run: int


def parse_record_timestamp(header: dict[str, Any]) -> datetime | None:
    """Parse the recorder's supported date and time formats."""
    date_text = str(header.get("Date", "")).strip()
    time_text = str(header.get("time", header.get("Time", ""))).strip()
    combined = f"{date_text} {time_text}".strip()
    candidates = [
        (combined, "%m/%d/%Y %H:%M:%S"),
        (combined, "%m/%d/%Y %H:%M"),
        (combined, "%Y-%m-%d %H:%M:%S"),
        (combined, "%Y-%m-%d %H:%M"),
        (date_text, "%m/%d/%Y"),
        (date_text, "%Y-%m-%d"),
        (date_text, "%m%d%Y"),
    ]
    for value, format_text in candidates:
        if not value:
            continue
        try:
            return datetime.strptime(value, format_text)
        except ValueError:
            continue
    return None


def calculate_run_pulses(
    pre_ablation: Any,
    ablation: Any,
    additional_on_target: Any,
    off_target: Any,
) -> tuple[int, int]:
    """Return on-target and all-laser pulse totals for one run."""
    on_target = sum(
        _pulse_count(value)
        for value in (pre_ablation, ablation, additional_on_target)
    )
    return on_target, on_target + _pulse_count(off_target)


def find_target_pulse_history(
    paths: Iterable[str | Path],
    *,
    before_timestamp: datetime,
    target_id: str = "",
    material: str = "",
    chamber: str = "",
) -> PulseHistoryResult:
    """Resolve target pulses from JSON records earlier than a timestamp.

    Exact Target ID records take priority. Material plus chamber is used only
    when no exact-ID history is available, which keeps legacy records usable.
    """
    normalized_target_id = _normalize(target_id)
    normalized_material = _normalize(material)
    normalized_chamber = _normalize(chamber)
    if not normalized_target_id and not normalized_material:
        raise ValueError("Enter a Target ID or Target Material before searching history.")

    exact_records: list[_PulseRecord] = []
    material_records: list[_PulseRecord] = []
    material_target_ids: set[str] = set()
    warnings: list[str] = []
    seen_paths: set[Path] = set()

    for input_path in paths:
        input_path = Path(input_path).expanduser().resolve()
        candidates = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.json"))
        for record_path in candidates:
            if record_path in seen_paths or not record_path.is_file():
                continue
            seen_paths.add(record_path)
            try:
                with record_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue

            header = record.get("header", {})
            if not isinstance(header, dict):
                continue
            timestamp = parse_record_timestamp(header)
            if timestamp is None or timestamp >= before_timestamp:
                continue
            record_chamber = _normalize(header.get("Chamber", ""))
            if normalized_chamber and record_chamber != normalized_chamber:
                continue

            growth_id = str(header.get("Growth ID", "")).strip()
            matching_material_sections = 0
            for section_name, section in record.items():
                if not str(section_name).startswith("target_") or not isinstance(section, dict):
                    continue
                record_target_id = _normalize(section.get("Target ID", ""))
                record_material = _normalize(section.get("Target Material", ""))
                pulse_record = _to_pulse_record(
                    section,
                    timestamp=timestamp,
                    path=record_path,
                    growth_id=growth_id,
                    target_id=record_target_id,
                    material=record_material,
                )
                if normalized_target_id and record_target_id == normalized_target_id:
                    exact_records.append(pulse_record)
                elif (
                    normalized_material
                    and record_material == normalized_material
                    and (not normalized_target_id or not record_target_id)
                ):
                    material_records.append(pulse_record)
                    matching_material_sections += 1
                    if record_target_id:
                        material_target_ids.add(record_target_id)

            if matching_material_sections > 1:
                raise ValueError(
                    f"{record_path.name} contains more than one '{material}' target. "
                    "Assign unique Target IDs before calculating history."
                )

    if not normalized_target_id and len(material_target_ids) > 1:
        raise ValueError(
            f"History contains multiple Target IDs for '{material}'. Enter the physical Target ID before calculating."
        )

    used_material_fallback = not exact_records
    records = exact_records or material_records
    if used_material_fallback and records:
        warnings.append("Matched legacy history by material and chamber; verify the physical target.")

    records = _deduplicate_records(records, warnings)
    records.sort(key=lambda item: (item.timestamp, str(item.path)))
    total = 0
    source: _PulseRecord | None = None
    for record in records:
        if record.after_pulses is not None:
            total = record.after_pulses
        elif record.before_pulses is not None:
            total = record.before_pulses + record.on_target_this_run
        else:
            total += record.on_target_this_run
        source = record

    return PulseHistoryResult(
        before_pulses=total,
        matching_records=len(records),
        source_path=source.path if source else None,
        source_growth_id=source.growth_id if source else "",
        source_timestamp=source.timestamp if source else None,
        used_material_fallback=used_material_fallback and bool(records),
        warning=" ".join(warnings),
    )


def _to_pulse_record(
    section: dict[str, Any],
    *,
    timestamp: datetime,
    path: Path,
    growth_id: str,
    target_id: str,
    material: str,
) -> _PulseRecord:
    on_target, _ = calculate_run_pulses(
        _first_value(section, "Pre-Ablation Pulses (count)", "Pre-Ablation-Pulses"),
        _first_value(section, "Ablation Pulses (count)", "Ablation-Pulses"),
        _first_value(section, "Additional On-Target Pulses (count)"),
        _first_value(section, "Off-Target Pulses (count)"),
    )
    return _PulseRecord(
        timestamp=timestamp,
        path=path,
        growth_id=growth_id,
        target_id=target_id,
        material=material,
        before_pulses=_optional_pulse_count(
            _first_value(section, "Target Pulses Before Run (count)")
        ),
        after_pulses=_optional_pulse_count(
            _first_value(
                section,
                "Target Pulses After Run (count)",
                "Cumulative Target Pulses (count)",
            )
        ),
        on_target_this_run=on_target,
    )


def _deduplicate_records(records: list[_PulseRecord], warnings: list[str]) -> list[_PulseRecord]:
    unique: dict[tuple[str, str], _PulseRecord] = {}
    for record in records:
        identity = record.target_id or record.material
        record_key = record.growth_id or str(record.path)
        key = (_normalize(record_key), identity)
        if key in unique:
            warnings.append(f"Ignored a duplicate copy of growth record '{record_key}'.")
            continue
        unique[key] = record
    return list(unique.values())


def _first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _optional_pulse_count(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _pulse_count(value)


def _pulse_count(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pulse count: {value!r}") from exc
    if numeric < 0 or not numeric.is_integer() or numeric > MAX_PULSE_COUNT:
        raise ValueError(f"Pulse count must be a whole number from 0 to {MAX_PULSE_COUNT:,}.")
    return int(numeric)
