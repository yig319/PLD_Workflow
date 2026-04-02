"""Small plotting and HDF5 helpers used by legacy plume notebooks."""

from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, labels=None, img_per_row: int = 8, colorbar: bool = False) -> None:
    """Display a list of 2-D images in a compact grid."""
    if len(images) == 0:
        raise ValueError("At least one image is required.")

    labels = list(labels) if labels is not None else list(range(len(images)))
    row_count = (len(images) + img_per_row - 1) // img_per_row
    aspect = images[0].shape[1] / max(images[0].shape[0], 1)
    fig, axes = plt.subplots(row_count, img_per_row, figsize=(16, max(2, row_count * (aspect + 1))))
    axes = np.atleast_2d(axes)

    for index in range(row_count * img_per_row):
        axis = axes[index // img_per_row, index % img_per_row]
        if index >= len(images):
            axis.axis("off")
            continue

        axis.set_title(str(labels[index]))
        image_handle = axis.imshow(images[index])
        axis.axis("off")
        if colorbar:
            fig.colorbar(image_handle, ax=axis)

    plt.tight_layout()
    plt.show()


def show_h5_dataset_name(ds_path, class_name=None) -> None:
    """Print the dataset names available in an HDF5 archive."""
    with h5py.File(ds_path) as handle:
        if class_name:
            print(list(handle[class_name].keys()))
        else:
            print(list(handle.keys()))


def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show: bool = True):
    """Load a plume dataset from HDF5 and optionally display example frames."""
    with h5py.File(ds_path) as handle:
        plumes = np.array(handle[class_name][ds_name])

    if show:
        images = process_func(plumes) if process_func else plumes
        show_images(images, colorbar=True)
    return plumes
