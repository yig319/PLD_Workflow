"""Utilities for archiving plume image folders and attaching PLD metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .parameter_export import build_default_file_stem


@dataclass
class PlumeTargetSummary:
    """Summary for one target packed into an HDF5 archive."""

    target_name: str
    plume_count: int
    frame_count: int
    frame_shape: tuple[int, int]


@dataclass
class PlumePackResult:
    """Summary returned after packing a plume directory into HDF5."""

    output_path: str
    target_summaries: list[PlumeTargetSummary]
    removed_ini_files: int

    @property
    def total_targets(self) -> int:
        """Return the number of packed target datasets."""
        return len(self.target_summaries)

    @property
    def total_plumes(self) -> int:
        """Return the total number of plume folders stored."""
        return sum(item.plume_count for item in self.target_summaries)

    @property
    def total_frames(self) -> int:
        """Return the total number of image frames stored."""
        return sum(item.frame_count for item in self.target_summaries)


def read_metadata_json(file_path: str | Path) -> dict[str, Any]:
    """Read a PLD metadata JSON file and return the decoded dictionary."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata_json(metadata: dict[str, Any], file_path: str | Path) -> None:
    """Write metadata to disk using a readable indented JSON layout."""
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def metadata_to_text(metadata: dict[str, Any]) -> str:
    """Format metadata for the editable plume-manager text box."""
    return json.dumps(metadata, indent=2, sort_keys=True)


def build_plume_archive_stem(source_dir: str | Path, metadata: dict[str, Any] | None = None) -> str:
    """Build a readable archive stem using metadata when available."""
    header = metadata.get("header", {}) if isinstance(metadata, dict) else {}
    if isinstance(header, dict):
        stem = build_default_file_stem(
            str(header.get("Growth ID", "")),
            str(header.get("User Name", "")),
            str(header.get("Date", "")),
        )
        if stem and not stem.startswith("growth_record_"):
            return f"{stem}_plume"
    return f"{Path(source_dir).resolve().name}_plume"


def pack_plume_directory(
    source_dir: str | Path,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> PlumePackResult:
    """Pack a plume dataset directory into a single HDF5 archive.

    The expected layout is:

    `source_dir/<target_name>/BMP/<plume_name>/<frame files>`

    The archive stores one dataset per target under `PLD_Plumes`.
    Each target dataset has shape `(num_plumes, max_frames, height, width)`.
    """
    h5py = _import_h5py()
    np = _import_numpy()

    source_root = Path(source_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Plume source directory does not exist: {source_root}")

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    removed_ini_files = remove_desktop_ini_files(source_root)
    target_summaries: list[PlumeTargetSummary] = []

    if output_file.exists():
        output_file.unlink()

    with h5py.File(output_file, "w") as handle:
        handle.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        handle.attrs["source_dir"] = str(source_root)
        if metadata:
            handle.attrs["metadata_json"] = metadata_to_text(metadata)

        plume_group = handle.create_group("PLD_Plumes")

        for target_dir in _iter_target_directories(source_root):
            plume_frames = list(_iter_plume_frame_groups(target_dir))
            if not plume_frames:
                continue

            first_frame = _read_frame(plume_frames[0][0])
            height, width = first_frame.shape
            frame_counts = np.array([len(group) for group in plume_frames], dtype=np.int32)
            max_frames = int(frame_counts.max()) if frame_counts.size else 0
            dataset = plume_group.create_dataset(
                target_dir.name,
                shape=(len(plume_frames), max_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
            )
            dataset.attrs["frame_counts"] = frame_counts
            dataset.attrs["source_target_dir"] = str(target_dir)

            total_frames = 0
            for plume_index, frame_group in enumerate(plume_frames):
                total_frames += len(frame_group)
                for frame_index, frame_path in enumerate(frame_group):
                    dataset[plume_index, frame_index] = _read_frame(frame_path)

            target_summaries.append(
                PlumeTargetSummary(
                    target_name=target_dir.name,
                    plume_count=len(plume_frames),
                    frame_count=total_frames,
                    frame_shape=(height, width),
                )
            )

        if not target_summaries:
            raise RuntimeError(
                "No plume frames were found. Expected folders like "
                "'<target>/BMP/<plume>/<frame files>'."
            )

    return PlumePackResult(
        output_path=str(output_file),
        target_summaries=target_summaries,
        removed_ini_files=removed_ini_files,
    )


def upload_archive_to_datafed(
    archive_path: str | Path,
    metadata: dict[str, Any],
    collection_id: str = "c/391937642",
):
    """Upload a packed HDF5 archive to DataFed with PLD metadata attached."""
    try:
        from datafed.CommandLib import API
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "DataFed is not available in this environment. Install the DataFed client before uploading."
        ) from exc

    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Packed archive was not found: {archive_file}")

    api = API()
    response = api.dataCreate(
        archive_file.stem,
        metadata=json.dumps(metadata),
        parent_id=collection_id,
    )
    record_id = response[0].data[0].id
    transfer = api.dataPut(record_id, str(archive_file), wait=True)
    return {"record_id": record_id, "transfer": transfer}


def remove_desktop_ini_files(root_dir: str | Path) -> int:
    """Remove stray Windows `desktop.ini` files below a plume dataset directory."""
    root = Path(root_dir)
    removed_count = 0
    for file_path in root.rglob("desktop.ini"):
        if file_path.is_file():
            file_path.unlink()
            removed_count += 1
    return removed_count


def _iter_target_directories(source_root: Path) -> Iterable[Path]:
    """Yield target directories stored directly below the plume source root."""
    for path in sorted(source_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def _iter_plume_frame_groups(target_dir: Path) -> Iterable[list[Path]]:
    """Yield sorted frame-file groups for each plume video folder in one target."""
    bmp_dir = target_dir / "BMP"
    if not bmp_dir.is_dir():
        return

    for plume_dir in sorted(path for path in bmp_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        frame_files = sorted(
            path
            for path in plume_dir.iterdir()
            if path.is_file() and path.name.lower() != "desktop.ini"
        )
        if frame_files:
            yield frame_files


def _read_frame(frame_path: Path):
    """Read one frame and normalize it into a 2-D uint8 array."""
    image = _import_matplotlib_image()
    np = _import_numpy()

    frame = image.imread(frame_path)
    array = np.asarray(frame)

    # Most plume frames are grayscale already. For RGB/RGBA images we average
    # channels so downstream metrics see one intensity image per frame.
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)

    if array.dtype.kind == "f":
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0

    array = np.asarray(array).clip(0, 255).astype(np.uint8)
    if array.ndim != 2:
        raise ValueError(f"Unsupported frame shape for {frame_path}: {array.shape}")
    return array


def _import_h5py():
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("h5py is required for plume archiving.") from exc
    return h5py


def _import_matplotlib_image():
    try:
        from matplotlib import image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for reading plume image frames.") from exc
    return image


def _import_numpy():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required for plume archiving.") from exc
    return np
