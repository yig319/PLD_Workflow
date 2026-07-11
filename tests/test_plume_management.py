from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from pld_workflow.plume_management import (
    build_plume_growth_stem,
    create_plume_workspace,
    inspect_plume_archive,
    pack_plume_directory,
    read_packed_frame,
    read_plume_frame,
    scan_plume_directory,
    stage_raw_files_for_target,
)


def _write_demo_frame(directory: Path, name: str, array) -> None:
    Image.fromarray(np.asarray(array)).save(directory / name)


def test_create_plume_workspace_creates_target_and_pre_ablation_folders(tmp_path):
    metadata = {
        "header": {"Growth ID": "demo"},
        "target_1": {
            "Target Material": "SrRuO3",
            "Pre-Ablation Pulses (count)": 50,
        },
        "target_2": {
            "Target Material": "LaAlO3",
            "Pre-Ablation Pulses (count)": 0,
        },
    }

    result = create_plume_workspace(tmp_path / "workspace", metadata)

    assert result.total_targets == 4
    created_names = [record.folder_name for record in result.target_folders]
    assert created_names == ["1-SrRuO3", "1-SrRuO3-Pre", "2-LaAlO3", "2-LaAlO3-Pre"]
    for record in result.target_folders:
        assert record.target_dir.is_dir()
        assert not (record.target_dir / "raw").exists()
        assert not (record.target_dir / "BMP").exists()


def test_build_plume_growth_stem_prefers_growth_id_when_available(tmp_path):
    metadata = {
        "header": {
            "Growth ID": "Growth1",
            "Sample Name": "SampleA",
            "User Name": "Person1",
            "Date": "04022026",
        }
    }

    assert build_plume_growth_stem(tmp_path, metadata=metadata) == "Growth1_Person1_04022026"


def test_stage_raw_files_for_target_moves_files_and_renames_conflicts(tmp_path):
    target_dir = tmp_path / "TargetA"
    target_dir.mkdir(parents=True)

    source_dir = tmp_path / "raw_inbox"
    source_dir.mkdir()
    file_one = source_dir / "capture.bin"
    file_two = source_dir / "capture_2.bin"
    file_one.write_bytes(b"one")
    file_two.write_bytes(b"two")
    (target_dir / "capture.bin").write_bytes(b"existing")

    result = stage_raw_files_for_target([file_one, file_two], target_dir)

    assert result.total_files == 2
    moved_names = [Path(path).name for path in result.moved_files]
    assert moved_names == ["capture_2.bin", "capture_2_2.bin"]
    assert not file_one.exists()
    assert not file_two.exists()


def test_scan_plume_directory_returns_nested_tree_without_removing_ini(tmp_path):
    source_root = tmp_path / "plume_dataset"
    plume_a = source_root / "TargetA" / "BMP" / "plume_001"
    plume_b = source_root / "TargetA" / "BMP" / "plume_002"
    plume_a.mkdir(parents=True)
    plume_b.mkdir(parents=True)

    _write_demo_frame(plume_a, "frame_002.png", np.array([[0, 1], [2, 3]], dtype=np.uint8))
    _write_demo_frame(plume_a, "frame_001.png", np.array([[4, 5], [6, 7]], dtype=np.uint8))
    _write_demo_frame(plume_b, "frame_001.png", np.array([[8, 9], [10, 11]], dtype=np.uint8))
    (plume_b / "desktop.ini").write_text("ignore me", encoding="utf-8")

    dataset = scan_plume_directory(source_root)

    assert dataset.total_targets == 1
    assert dataset.total_plumes == 2
    assert dataset.total_frames == 3
    assert dataset.removed_ini_files == 0
    assert dataset.targets[0].name == "TargetA"
    assert [folder.name for folder in dataset.targets[0].plume_folders] == ["plume_001", "plume_002"]
    assert [frame.name for frame in dataset.targets[0].plume_folders[0].frames] == [
        "frame_001.png",
        "frame_002.png",
    ]
    assert (plume_b / "desktop.ini").exists()


def test_scan_plume_directory_can_remove_desktop_ini_files(tmp_path):
    source_root = tmp_path / "plume_dataset"
    plume_dir = source_root / "TargetA" / "BMP" / "plume_001"
    plume_dir.mkdir(parents=True)
    _write_demo_frame(plume_dir, "frame_001.png", np.array([[0, 1], [2, 3]], dtype=np.uint8))
    ini_path = plume_dir / "desktop.ini"
    ini_path.write_text("ignore me", encoding="utf-8")

    dataset = scan_plume_directory(source_root, remove_ini_files_first=True)

    assert dataset.removed_ini_files == 1
    assert not ini_path.exists()


def test_scan_plume_directory_accepts_frames_stored_directly_under_bmp(tmp_path):
    source_root = tmp_path / "plume_dataset"
    bmp_dir = source_root / "TargetA" / "BMP"
    bmp_dir.mkdir(parents=True)

    _write_demo_frame(bmp_dir, "frame_002.png", np.array([[0, 1], [2, 3]], dtype=np.uint8))
    _write_demo_frame(bmp_dir, "frame_001.png", np.array([[4, 5], [6, 7]], dtype=np.uint8))

    dataset = scan_plume_directory(source_root)

    assert dataset.total_targets == 1
    assert dataset.total_plumes == 1
    assert dataset.total_frames == 2
    assert dataset.targets[0].plume_folders[0].name == "BMP_root"
    assert [frame.name for frame in dataset.targets[0].plume_folders[0].frames] == [
        "frame_001.png",
        "frame_002.png",
    ]


def test_read_plume_frame_converts_rgb_data_to_grayscale_uint8(tmp_path):
    frame_path = tmp_path / "frame_rgb.png"
    rgb_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_frame[..., 0] = 255
    Image.fromarray(rgb_frame).save(frame_path)

    frame = read_plume_frame(frame_path)

    assert frame.shape == (2, 2)
    assert frame.dtype == np.uint8


def test_pack_plume_directory_creates_expected_hdf5_layout(tmp_path):
    source_root = tmp_path / "plume_dataset"
    plume_dir = source_root / "TargetA" / "BMP" / "plume_001"
    plume_dir.mkdir(parents=True)

    frame_1 = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    frame_2 = np.array([[40, 50], [60, 70]], dtype=np.uint8)
    _write_demo_frame(plume_dir, "frame_001.png", frame_1)
    _write_demo_frame(plume_dir, "frame_002.png", frame_2)

    output_path = tmp_path / "packed_plume.h5"
    result = pack_plume_directory(source_root, output_path, metadata={"header": {"Growth ID": "demo"}})

    assert result.total_targets == 1
    assert result.total_plumes == 1
    assert result.total_frames == 2

    with h5py.File(output_path, "r") as handle:
        assert "PLD_Plumes" in handle
        dataset = handle["PLD_Plumes"]["TargetA"]
        assert dataset.shape == (1, 2, 2, 2)
        assert dataset.attrs["frame_counts"][0] == 2


def test_pack_plume_directory_accepts_frames_stored_directly_under_bmp(tmp_path):
    source_root = tmp_path / "plume_dataset"
    bmp_dir = source_root / "TargetA" / "BMP"
    bmp_dir.mkdir(parents=True)

    frame_1 = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    frame_2 = np.array([[40, 50], [60, 70]], dtype=np.uint8)
    _write_demo_frame(bmp_dir, "frame_001.png", frame_1)
    _write_demo_frame(bmp_dir, "frame_002.png", frame_2)

    output_path = tmp_path / "packed_plume.h5"
    result = pack_plume_directory(source_root, output_path, metadata={"header": {"Growth ID": "demo"}})

    assert result.total_targets == 1
    assert result.total_plumes == 1
    assert result.total_frames == 2

    with h5py.File(output_path, "r") as handle:
        dataset = handle["PLD_Plumes"]["TargetA"]
        assert dataset.shape == (1, 2, 2, 2)
        assert dataset.attrs["frame_counts"][0] == 2


def test_inspect_plume_archive_and_read_packed_frame_round_trip(tmp_path):
    source_root = tmp_path / "plume_dataset"
    plume_dir = source_root / "TargetA" / "BMP" / "plume_001"
    plume_dir.mkdir(parents=True)

    frame_1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    frame_2 = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    _write_demo_frame(plume_dir, "frame_001.png", frame_1)
    _write_demo_frame(plume_dir, "frame_002.png", frame_2)

    output_path = tmp_path / "packed_plume.h5"
    pack_plume_directory(source_root, output_path, metadata={"header": {"Growth ID": "demo"}})

    archive_record = inspect_plume_archive(output_path)
    loaded_frame = read_packed_frame(output_path, "TargetA", 0, 1)

    assert archive_record.total_targets == 1
    assert archive_record.total_plumes == 1
    assert archive_record.total_frames == 2
    assert archive_record.targets[0].frame_counts == [2]
    assert loaded_frame.shape == (2, 2)
    assert loaded_frame.dtype == np.uint8
