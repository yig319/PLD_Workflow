"""Backward-compatible wrapper for `pld_workflow.plume_management`."""

from pld_workflow.plume_management import (
    PlumePackResult,
    PlumeTargetSummary,
    build_plume_archive_stem,
    metadata_to_text,
    pack_plume_directory,
    read_metadata_json,
    remove_desktop_ini_files,
    upload_archive_to_datafed,
    write_metadata_json,
)

__all__ = [
    "PlumePackResult",
    "PlumeTargetSummary",
    "build_plume_archive_stem",
    "metadata_to_text",
    "pack_plume_directory",
    "read_metadata_json",
    "remove_desktop_ini_files",
    "upload_archive_to_datafed",
    "write_metadata_json",
]
