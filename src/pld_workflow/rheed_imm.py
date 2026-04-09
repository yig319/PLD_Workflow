"""Lightweight IMM readers used by the standalone RHEED frame visualizer.

The logic here is adapted from the notebook workflow in the sibling
`RHEED_RealTimeAnalyzer` repository, but trimmed down for the common
desktop use case in this project:

- inspect one `.imm` file without loading the whole movie into memory
- convert user timing inputs into a single frame index
- load one frame directly from disk on demand
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ImmInfo:
    """Basic metadata inferred from a k-Space IMM movie file.

    Parameters
    ----------
    path:
        Absolute path to the inspected `.imm` file.
    frame_count:
        Number of complete frames found using the configured frame stride.
    height, width:
        Pixel dimensions of each frame in the raw detector layout.
    dtype:
        NumPy dtype string used to decode the frame payload.
    frame_stride_bytes:
        Number of bytes occupied by one IMM frame block, including header.
    header_bytes:
        Number of bytes at the start of each frame block that store metadata.
    signature:
        Short marker expected in the first frame header.
    fps_estimate:
        Optional frames-per-second value supplied by the caller. IMM files do
        not usually store acquisition fps directly, so this is user-provided.
    trailing_bytes:
        Remaining bytes at the end of the file after complete frame blocks.
    """

    path: Path
    frame_count: int
    height: int
    width: int
    dtype: str
    frame_stride_bytes: int
    header_bytes: int
    signature: str
    fps_estimate: float | None = None
    trailing_bytes: int = 0


def inspect_imm_file(
    path: str | Path,
    *,
    frame_stride_bytes: int = 646_144,
    header_bytes: int = 640,
    width: int = 656,
    height: int = 492,
    dtype: str = "<u2",
    signature: bytes = b"KSA00F",
    fps: float | None = None,
) -> ImmInfo:
    """Inspect a k-Space IMM file using its fixed frame layout.

    Notes
    -----
    The defaults match the layout used in the RHEED notebook demo. The function
    reads only the file size plus the first frame header, so it stays fast and
    memory-light even for large movies.
    """

    file_path = Path(path).expanduser().resolve()
    size_bytes = file_path.stat().st_size
    if size_bytes < header_bytes:
        raise ValueError("IMM file is smaller than one frame header.")
    if frame_stride_bytes <= header_bytes:
        raise ValueError("frame_stride_bytes must be greater than header_bytes.")

    pixel_dtype = np.dtype(dtype)
    payload_bytes = frame_stride_bytes - header_bytes
    expected_payload_bytes = pixel_dtype.itemsize * width * height
    if payload_bytes != expected_payload_bytes:
        raise ValueError(
            "IMM payload size does not match dtype * width * height: "
            f"{payload_bytes} != {expected_payload_bytes}"
        )

    trailing_bytes = int(size_bytes % frame_stride_bytes)
    frame_count = int(size_bytes // frame_stride_bytes)
    if frame_count <= 0:
        raise ValueError("IMM file does not contain one complete frame.")

    with file_path.open("rb") as handle:
        first_header = handle.read(header_bytes)
    if signature not in first_header:
        raise ValueError(f"IMM signature {signature!r} was not found in the first header.")

    fps_value = None if fps is None else float(fps)
    return ImmInfo(
        path=file_path,
        frame_count=frame_count,
        height=int(height),
        width=int(width),
        dtype=pixel_dtype.str,
        frame_stride_bytes=int(frame_stride_bytes),
        header_bytes=int(header_bytes),
        signature=signature.decode("ascii", errors="replace"),
        fps_estimate=fps_value,
        trailing_bytes=trailing_bytes,
    )


class ImmMovie:
    """Memory-light interface for working with one IMM movie on disk.

    Parameters
    ----------
    path:
        File to inspect and sample from.
    fps:
        User-supplied movie frame rate in frames/second. This is required when
        converting from elapsed time to frame index.
    frame_stride_bytes, header_bytes, width, height, dtype, signature:
        Low-level IMM layout settings. The defaults match the existing notebook
        workflow and common k-Space IMM exports.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        frame_stride_bytes: int = 646_144,
        header_bytes: int = 640,
        width: int = 656,
        height: int = 492,
        dtype: str = "<u2",
        signature: bytes = b"KSA00F",
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.frame_stride_bytes = int(frame_stride_bytes)
        self.header_bytes = int(header_bytes)
        self.width = int(width)
        self.height = int(height)
        self.dtype = np.dtype(dtype).str
        self.signature = bytes(signature)
        self.info = inspect_imm_file(
            self.path,
            fps=fps,
            frame_stride_bytes=self.frame_stride_bytes,
            header_bytes=self.header_bytes,
            width=self.width,
            height=self.height,
            dtype=self.dtype,
            signature=self.signature,
        )
        self.fps = None if fps is None else float(fps)

    @property
    def frame_count(self) -> int:
        """Return the number of complete frames in the file."""

        return int(self.info.frame_count)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the detector frame shape as `(height, width)`."""

        return int(self.info.height), int(self.info.width)

    @property
    def duration_s(self) -> float | None:
        """Return the movie duration in seconds when fps is known."""

        if self.fps is None or self.fps <= 0:
            return None
        return float(self.frame_count / self.fps)

    def inspect(self) -> ImmInfo:
        """Return cached metadata for the current movie."""

        return self.info

    def frame_index_from_time(self, time_s: float) -> int:
        """Convert elapsed time in seconds to the nearest frame index."""

        if self.fps is None or self.fps <= 0:
            raise ValueError("Movie fps must be set before loading by time.")
        raw_index = int(round(float(time_s) * float(self.fps)))
        return int(np.clip(raw_index, 0, self.frame_count - 1))

    def time_from_frame_index(self, frame_index: int) -> float:
        """Convert one frame index back to elapsed time in seconds."""

        if self.fps is None or self.fps <= 0:
            raise ValueError("Movie fps must be set before converting frame index to time.")
        return float(frame_index) / float(self.fps)

    def time_from_pulse_count(self, pulse_count: float, laser_rate_hz: float) -> float:
        """Convert laser pulse count to elapsed time in seconds.

        `laser_rate_hz` is the laser repetition rate in pulses/second.
        """

        if laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be greater than zero.")
        return float(pulse_count) / float(laser_rate_hz)

    def frame_index_from_pulse_count(self, pulse_count: float, laser_rate_hz: float) -> int:
        """Convert pulse count to a frame index using laser rate and movie fps."""

        return self.frame_index_from_time(self.time_from_pulse_count(pulse_count, laser_rate_hz))

    def pulse_count_from_frame_index(self, frame_index: int, laser_rate_hz: float) -> float:
        """Estimate how many laser pulses had occurred by one frame index."""

        if laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be greater than zero.")
        return float(self.time_from_frame_index(frame_index) * float(laser_rate_hz))

    def load_frame_raw(self, frame_index: int) -> np.ndarray:
        """Read one raw frame directly from disk without loading the full movie."""

        if frame_index < 0 or frame_index >= self.frame_count:
            raise ValueError(f"frame_index must be between 0 and {self.frame_count - 1}.")

        pixel_dtype = np.dtype(self.dtype)
        payload_bytes = pixel_dtype.itemsize * self.width * self.height
        with self.path.open("rb") as handle:
            offset = int(frame_index) * self.frame_stride_bytes + self.header_bytes
            handle.seek(offset)
            payload = handle.read(payload_bytes)

        if len(payload) != payload_bytes:
            raise ValueError(f"Incomplete IMM frame payload at index {frame_index}.")
        return np.frombuffer(payload, dtype=pixel_dtype).reshape(self.height, self.width)

    def load_frame(self, frame_index: int, *, as_float: bool = True) -> np.ndarray:
        """Load one frame from disk, optionally converting it to float."""

        frame = self.load_frame_raw(frame_index)
        return np.asarray(frame, dtype=float) if as_float else frame

    def load_frame_by_time(self, time_s: float, *, as_float: bool = True) -> tuple[int, np.ndarray]:
        """Load the frame nearest to the requested time."""

        frame_index = self.frame_index_from_time(time_s)
        return frame_index, self.load_frame(frame_index, as_float=as_float)

    def load_frame_by_pulse_count(
        self,
        pulse_count: float,
        *,
        laser_rate_hz: float,
        as_float: bool = True,
    ) -> tuple[int, np.ndarray]:
        """Load the frame nearest to the requested laser pulse count."""

        frame_index = self.frame_index_from_pulse_count(pulse_count, laser_rate_hz)
        return frame_index, self.load_frame(frame_index, as_float=as_float)


__all__ = ["ImmInfo", "ImmMovie", "inspect_imm_file"]
