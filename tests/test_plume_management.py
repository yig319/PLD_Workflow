from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
plt = pytest.importorskip("matplotlib.pyplot")
np = pytest.importorskip("numpy")

from pld_workflow.plume_management import pack_plume_directory


def test_pack_plume_directory_creates_expected_hdf5_layout(tmp_path):
    source_root = tmp_path / "plume_dataset"
    plume_dir = source_root / "TargetA" / "BMP" / "plume_001"
    plume_dir.mkdir(parents=True)

    frame_1 = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    frame_2 = np.array([[40, 50], [60, 70]], dtype=np.uint8)
    plt.imsave(plume_dir / "frame_001.png", frame_1, cmap="gray")
    plt.imsave(plume_dir / "frame_002.png", frame_2, cmap="gray")

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
