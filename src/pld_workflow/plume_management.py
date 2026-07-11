"""Utilities for inspecting and archiving plume image folders."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .parameter_export import build_default_file_stem


FRAME_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass
class PlumeWorkspaceTarget:
    """One target folder derived from metadata for the plume workspace."""

    section_key: str
    folder_name: str
    display_name: str
    source_target_name: str
    is_pre_ablation: bool
    pre_ablation_pulse_count: int


@dataclass
class PlumeWorkspaceFolderRecord:
    """One created target folder inside a plume workspace."""

    section_key: str
    folder_name: str
    target_dir: Path
    is_pre_ablation: bool


@dataclass
class PlumeWorkspaceCreationResult:
    """Summary returned after creating a workspace from metadata."""

    root_dir: str
    target_folders: list[PlumeWorkspaceFolderRecord]

    @property
    def total_targets(self) -> int:
        """Return the number of target folders created."""
        return len(self.target_folders)


@dataclass
class RawFileStagingResult:
    """Summary returned after moving raw files into one target folder."""

    target_dir: str
    destination_dir: str
    moved_files: list[str]

    @property
    def total_files(self) -> int:
        """Return the number of moved files."""
        return len(self.moved_files)


@dataclass
class PlumeFrameRecord:
    """One frame file inside a plume folder."""

    path: Path

    @property
    def name(self) -> str:
        """Return the frame file name."""
        return self.path.name


@dataclass
class PlumeFolderRecord:
    """One plume folder containing a time-ordered list of frames."""

    name: str
    directory: Path
    frames: list[PlumeFrameRecord]

    @property
    def frame_count(self) -> int:
        """Return the number of frame files inside this plume folder."""
        return len(self.frames)


@dataclass
class PlumeTargetRecord:
    """One target directory containing one or more plume folders."""

    name: str
    directory: Path
    plume_folders: list[PlumeFolderRecord]

    @property
    def plume_count(self) -> int:
        """Return the number of plume folders below this target."""
        return len(self.plume_folders)

    @property
    def frame_count(self) -> int:
        """Return the number of frames across all plume folders."""
        return sum(folder.frame_count for folder in self.plume_folders)


@dataclass
class PlumeDatasetRecord:
    """Structured description of a plume directory tree."""

    source_root: Path
    targets: list[PlumeTargetRecord]
    removed_ini_files: int = 0

    @property
    def total_targets(self) -> int:
        """Return the number of target directories with plume data."""
        return len(self.targets)

    @property
    def total_plumes(self) -> int:
        """Return the number of plume folders across all targets."""
        return sum(target.plume_count for target in self.targets)

    @property
    def total_frames(self) -> int:
        """Return the number of frames across the full dataset."""
        return sum(target.frame_count for target in self.targets)


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


@dataclass
class PlumeArchiveTargetRecord:
    """One packed target dataset stored inside an HDF5 archive."""

    target_name: str
    plume_count: int
    frame_counts: list[int]
    frame_shape: tuple[int, int]

    @property
    def total_frames(self) -> int:
        """Return the total number of stored frames for this target."""
        return sum(self.frame_counts)


@dataclass
class PlumeArchiveRecord:
    """Structured description of one packed plume HDF5 archive."""

    archive_path: str
    source_dir: str
    created_at: str
    metadata_json: str | None
    targets: list[PlumeArchiveTargetRecord]

    @property
    def total_targets(self) -> int:
        """Return the number of packed target datasets."""
        return len(self.targets)

    @property
    def total_plumes(self) -> int:
        """Return the number of plume folders represented in the archive."""
        return sum(target.plume_count for target in self.targets)

    @property
    def total_frames(self) -> int:
        """Return the number of stored frames across the whole archive."""
        return sum(target.total_frames for target in self.targets)


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


def build_plume_growth_stem(source_dir: str | Path, metadata: dict[str, Any] | None = None) -> str:
    """Build the default growth-folder stem using metadata when available."""

    header = metadata.get("header", {}) if isinstance(metadata, dict) else {}
    if isinstance(header, dict):
        stem = build_default_file_stem(
            _preferred_growth_name(header),
            str(header.get("User Name", "")),
            str(header.get("Date", "")),
        )
        if stem and not stem.startswith("growth_record_"):
            return stem
    return f"{Path(source_dir).resolve().name}"


def build_plume_archive_stem(source_dir: str | Path, metadata: dict[str, Any] | None = None) -> str:
    """Build a readable archive stem using metadata when available."""
    return f"{build_plume_growth_stem(source_dir, metadata=metadata)}_plume"


def build_plume_workspace_targets(
    metadata: dict[str, Any],
    *,
    include_pre_ablation: bool = True,
) -> list[PlumeWorkspaceTarget]:
    """Build target-folder definitions from a PLD metadata JSON object."""

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    targets: list[PlumeWorkspaceTarget] = []
    seen_names: set[str] = set()
    for section_key, target_data in _iter_metadata_targets(metadata):
        raw_name = str(target_data.get("Target Material", "")).strip() or section_key
        target_index = _target_index_label(section_key)
        base_name = _deduplicate_workspace_name(
            _workspace_target_base_name(target_index, raw_name, fallback=section_key),
            seen_names,
        )
        seen_names.add(base_name)
        pre_ablation_pulse_count = _coerce_non_negative_int(
            target_data.get("Pre-Ablation Pulses (count)", 0)
        )

        targets.append(
            PlumeWorkspaceTarget(
                section_key=section_key,
                folder_name=base_name,
                display_name=raw_name,
                source_target_name=base_name,
                is_pre_ablation=False,
                pre_ablation_pulse_count=pre_ablation_pulse_count,
            )
        )

        if include_pre_ablation:
            pre_name = _deduplicate_workspace_name(_pre_ablation_folder_name(base_name), seen_names)
            seen_names.add(pre_name)
            targets.append(
                PlumeWorkspaceTarget(
                    section_key=section_key,
                    folder_name=pre_name,
                    display_name=f"{raw_name} Pre-Ablation",
                    source_target_name=base_name,
                    is_pre_ablation=True,
                    pre_ablation_pulse_count=pre_ablation_pulse_count,
                )
            )

    return targets


def create_plume_workspace(
    root_dir: str | Path,
    metadata: dict[str, Any],
    *,
    include_pre_ablation: bool = True,
) -> PlumeWorkspaceCreationResult:
    """Create target workspace folders from PLD metadata.

    The created layout is:

    `root_dir/<target_name>`

    Raw files can be moved directly into each target folder. Downstream decoder
    software is expected to create the `BMP` folder later when images are decoded.
    """

    root = Path(root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    targets = build_plume_workspace_targets(
        metadata,
        include_pre_ablation=include_pre_ablation,
    )
    if not targets:
        raise ValueError("Metadata does not contain any target sections like 'target_1'.")

    folder_records: list[PlumeWorkspaceFolderRecord] = []
    for target in targets:
        target_dir = root / target.folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        folder_records.append(
            PlumeWorkspaceFolderRecord(
                section_key=target.section_key,
                folder_name=target.folder_name,
                target_dir=target_dir,
                is_pre_ablation=target.is_pre_ablation,
            )
        )

    return PlumeWorkspaceCreationResult(
        root_dir=str(root),
        target_folders=folder_records,
    )


def scan_plume_directory(
    source_dir: str | Path,
    *,
    remove_ini_files_first: bool = False,
) -> PlumeDatasetRecord:
    """Inspect a plume dataset directory and return its nested structure.

    The expected layout is:

    `source_dir/<target_name>/BMP/<plume_name>/<frame files>`
    or
    `source_dir/<target_name>/BMP/<frame files>`
    """

    source_root = Path(source_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Plume source directory does not exist: {source_root}")

    removed_ini_files = 0
    if remove_ini_files_first:
        removed_ini_files = remove_desktop_ini_files(source_root)

    targets: list[PlumeTargetRecord] = []
    for target_dir in _iter_target_directories(source_root):
        plume_folders = _collect_plume_folders(target_dir)
        if plume_folders:
            targets.append(
                PlumeTargetRecord(
                    name=target_dir.name,
                    directory=target_dir,
                    plume_folders=plume_folders,
                )
            )

    return PlumeDatasetRecord(
        source_root=source_root,
        targets=targets,
        removed_ini_files=removed_ini_files,
    )


def stage_raw_files_for_target(
    source_paths: list[str | Path],
    target_dir: str | Path,
) -> RawFileStagingResult:
    """Move raw files directly into the selected target folder."""

    target_root = Path(target_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    moved_files: list[str] = []
    for source_path in source_paths:
        source_file = Path(source_path).expanduser().resolve()
        if not source_file.is_file():
            raise FileNotFoundError(f"Raw source file was not found: {source_file}")

        destination = _deduplicate_path(target_root / source_file.name)
        shutil.move(str(source_file), str(destination))
        moved_files.append(str(destination))

    return RawFileStagingResult(
        target_dir=str(target_root),
        destination_dir=str(target_root),
        moved_files=moved_files,
    )


def pack_plume_directory(
    source_dir: str | Path,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> PlumePackResult:
    """Pack a plume dataset directory into a single HDF5 archive.

    The expected layout is:

    `source_dir/<target_name>/BMP/<plume_name>/<frame files>`
    or
    `source_dir/<target_name>/BMP/<frame files>`

    The archive stores one dataset per target under `PLD_Plumes`.
    Each target dataset has shape `(num_plumes, max_frames, height, width)`.
    """
    h5py = _import_h5py()
    np = _import_numpy()

    dataset_record = scan_plume_directory(source_dir, remove_ini_files_first=True)
    source_root = dataset_record.source_root
    if not dataset_record.targets:
        raise RuntimeError(
            "No plume frames were found. Expected folders like "
            "'<target>/BMP/<plume>/<frame files>' or '<target>/BMP/<frame files>'."
        )

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_summaries: list[PlumeTargetSummary] = []

    if output_file.exists():
        output_file.unlink()

    with h5py.File(output_file, "w") as handle:
        handle.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        handle.attrs["source_dir"] = str(source_root)
        if metadata:
            handle.attrs["metadata_json"] = metadata_to_text(metadata)

        plume_group = handle.create_group("PLD_Plumes")

        for target in dataset_record.targets:
            first_frame = read_plume_frame(target.plume_folders[0].frames[0].path)
            height, width = first_frame.shape
            frame_counts = np.array(
                [folder.frame_count for folder in target.plume_folders],
                dtype=np.int32,
            )
            max_frames = int(frame_counts.max()) if frame_counts.size else 0
            dataset = plume_group.create_dataset(
                target.name,
                shape=(len(target.plume_folders), max_frames, height, width),
                dtype=np.uint8,
                fillvalue=0,
            )
            dataset.attrs["frame_counts"] = frame_counts
            dataset.attrs["source_target_dir"] = str(target.directory)

            total_frames = 0
            for plume_index, plume_folder in enumerate(target.plume_folders):
                total_frames += plume_folder.frame_count
                for frame_index, frame_record in enumerate(plume_folder.frames):
                    dataset[plume_index, frame_index] = read_plume_frame(frame_record.path)

            target_summaries.append(
                PlumeTargetSummary(
                    target_name=target.name,
                    plume_count=target.plume_count,
                    frame_count=total_frames,
                    frame_shape=(height, width),
                )
            )

    return PlumePackResult(
        output_path=str(output_file),
        target_summaries=target_summaries,
        removed_ini_files=dataset_record.removed_ini_files,
    )


def remove_desktop_ini_files(root_dir: str | Path) -> int:
    """Remove stray Windows `desktop.ini` files below a plume dataset directory."""
    root = Path(root_dir)
    removed_count = 0
    for file_path in root.rglob("desktop.ini"):
        if file_path.is_file():
            file_path.unlink()
            removed_count += 1
    return removed_count


def read_plume_frame(frame_path: str | Path):
    """Read one plume frame and normalize it into a 2-D uint8 array."""
    return _read_frame(Path(frame_path))


def inspect_plume_archive(archive_path: str | Path) -> PlumeArchiveRecord:
    """Inspect one packed plume HDF5 archive and summarize its contents."""

    h5py = _import_h5py()
    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Packed archive was not found: {archive_file}")

    with h5py.File(archive_file, "r") as handle:
        if "PLD_Plumes" not in handle:
            raise KeyError(f"{archive_file} does not contain a 'PLD_Plumes' group.")

        plume_group = handle["PLD_Plumes"]
        target_records: list[PlumeArchiveTargetRecord] = []
        for target_name in sorted(plume_group.keys()):
            dataset = plume_group[target_name]
            if dataset.ndim != 4:
                raise ValueError(
                    f"Packed dataset '{target_name}' has unsupported shape {dataset.shape}."
                )

            raw_frame_counts = dataset.attrs.get("frame_counts")
            if raw_frame_counts is None:
                frame_counts = [int(dataset.shape[1])] * int(dataset.shape[0])
            else:
                frame_counts = [int(value) for value in raw_frame_counts]

            target_records.append(
                PlumeArchiveTargetRecord(
                    target_name=target_name,
                    plume_count=int(dataset.shape[0]),
                    frame_counts=frame_counts,
                    frame_shape=(int(dataset.shape[2]), int(dataset.shape[3])),
                )
            )

        return PlumeArchiveRecord(
            archive_path=str(archive_file),
            source_dir=str(handle.attrs.get("source_dir", "")),
            created_at=str(handle.attrs.get("created_at", "")),
            metadata_json=(
                str(handle.attrs["metadata_json"])
                if "metadata_json" in handle.attrs
                else None
            ),
            targets=target_records,
        )


def read_packed_frame(
    archive_path: str | Path,
    target_name: str,
    plume_index: int,
    frame_index: int,
):
    """Load one frame from a packed plume HDF5 archive."""

    h5py = _import_h5py()
    np = _import_numpy()

    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Packed archive was not found: {archive_file}")

    with h5py.File(archive_file, "r") as handle:
        try:
            dataset = handle["PLD_Plumes"][target_name]
        except KeyError as exc:
            raise KeyError(f"Target '{target_name}' was not found in {archive_file}.") from exc

        frame_counts = [int(value) for value in dataset.attrs.get("frame_counts", [])]
        plume_count = int(dataset.shape[0])
        if plume_index < 0 or plume_index >= plume_count:
            raise IndexError(f"Plume index {plume_index} is outside 0..{plume_count - 1}.")

        available_frames = frame_counts[plume_index] if plume_index < len(frame_counts) else int(dataset.shape[1])
        if frame_index < 0 or frame_index >= available_frames:
            raise IndexError(
                f"Frame index {frame_index} is outside 0..{available_frames - 1} for target '{target_name}'."
            )

        return np.asarray(dataset[plume_index, frame_index]).astype(np.uint8)


def _iter_target_directories(source_root: Path):
    """Yield target directories stored directly below the plume source root."""
    for path in sorted(source_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def _collect_plume_folders(target_dir: Path) -> list[PlumeFolderRecord]:
    """Return the plume folders found below one target directory."""
    bmp_dir = target_dir / "BMP"
    if not bmp_dir.is_dir():
        return []

    plume_folders: list[PlumeFolderRecord] = []
    direct_bmp_frames = _collect_frame_records(bmp_dir)
    if direct_bmp_frames:
        plume_folders.append(
            PlumeFolderRecord(
                name="BMP_root",
                directory=bmp_dir,
                frames=direct_bmp_frames,
            )
        )

    for plume_dir in sorted(path for path in bmp_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        frame_records = _collect_frame_records(plume_dir)
        if frame_records:
            plume_folders.append(
                PlumeFolderRecord(
                    name=plume_dir.name,
                    directory=plume_dir,
                    frames=sorted(frame_records, key=lambda record: record.path.name.lower()),
                )
            )
    return plume_folders


def _collect_frame_records(directory: Path) -> list[PlumeFrameRecord]:
    """Return sorted frame files from one directory, ignoring non-image files."""

    return sorted(
        [
            PlumeFrameRecord(path)
            for path in directory.iterdir()
            if path.is_file()
            and path.name.lower() != "desktop.ini"
            and path.suffix.lower() in FRAME_SUFFIXES
        ],
        key=lambda record: record.path.name.lower(),
    )


def _read_frame(frame_path: Path):
    """Read one frame and normalize it into a 2-D uint8 array."""
    pil_image = _import_pillow_image()
    np = _import_numpy()

    with pil_image.open(frame_path) as frame:
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


def _iter_metadata_targets(metadata: dict[str, Any]):
    """Yield ordered `(section_key, target_dict)` pairs from metadata."""

    target_keys = [key for key in metadata.keys() if key.lower().startswith("target_")]
    target_keys.sort(key=lambda key: int(key.split("_")[1]) if key.split("_")[1].isdigit() else key)
    for key in target_keys:
        target_data = metadata.get(key)
        if isinstance(target_data, dict):
            yield key, target_data


def _preferred_growth_name(header: dict[str, Any]) -> str:
    """Return the header field preferred for naming the growth folder/archive."""

    growth_id = str(header.get("Growth ID", "")).strip()
    if growth_id:
        return growth_id
    return str(header.get("Sample Name", "")).strip()


def _target_index_label(section_key: str) -> str:
    """Return the display index derived from a metadata target section key."""

    suffix = section_key.split("_")[-1]
    return suffix if suffix.isdigit() else section_key


def _workspace_target_base_name(index_label: str, raw_name: str, *, fallback: str) -> str:
    """Build the base target folder name as `index-target` unless already prefixed."""

    sanitized_name = _sanitize_workspace_name(raw_name, fallback=fallback)
    lowered = sanitized_name.lower()
    if lowered.startswith(f"{index_label.lower()}-") or lowered.startswith(f"{index_label.lower()}_"):
        return sanitized_name
    return f"{index_label}-{sanitized_name}"


def _pre_ablation_folder_name(base_name: str) -> str:
    """Return the matching pre-ablation folder name for one target folder."""

    lowered = base_name.lower()
    if lowered.endswith("-pre") or lowered.endswith("_pre"):
        return base_name
    return f"{base_name}-Pre"


def _sanitize_workspace_name(name: str, *, fallback: str) -> str:
    """Convert a user-facing target name into a filesystem-safe folder name."""

    candidate = (name or "").strip()
    if not candidate:
        candidate = fallback
    candidate = re.sub(r"\s+", "_", candidate)
    candidate = re.sub(r'[<>:"/\\\\|?*]+', "_", candidate)
    candidate = candidate.strip(" ._")
    return candidate or fallback


def _deduplicate_workspace_name(name: str, seen_names: set[str]) -> str:
    """Return a unique workspace folder name within one target list."""

    if name not in seen_names:
        return name

    index = 2
    while f"{name}_{index}" in seen_names:
        index += 1
    return f"{name}_{index}"


def _deduplicate_path(path: Path) -> Path:
    """Return a non-conflicting destination path for one moved file."""

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _coerce_non_negative_int(value: Any) -> int:
    """Convert numeric-like values into a non-negative integer."""

    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _import_h5py():
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("h5py is required for plume archiving.") from exc
    return h5py


def _import_pillow_image():
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for reading plume image frames.") from exc
    return Image


def _import_numpy():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required for plume archiving.") from exc
    return np
