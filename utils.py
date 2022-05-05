import h5py 
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, labels=None, img_per_row=8, colorbar=False):

    '''
    This is a utility function used to show a series images.

    :param images: input images
    :type images: np.array

    :param labels: labels for images
    :type labels: str(, optional)

    :param img_per_row: how many images to show in one row
    :type img_per_row: int(, optional)

    :param colorbar: to determine if colobar is included 
    :type colorbar: bool(, optional)
    '''

    h = images[0].shape[1] // images[0].shape[0]*0.5 + 1
    if not labels:
        labels = range(len(images))
    fig, axes = plt.subplots(len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, h*len(images)//img_per_row+1))
    for i in range(len(images)):
        if len(images) <= img_per_row:
            axes[i%img_per_row].title.set_text(labels[i])
            im = axes[i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i%img_per_row])
            axes[i//img_per_row, i%img_per_row].axis('off')

        else:
            axes[i//img_per_row, i%img_per_row].title.set_text(labels[i])
            im = axes[i//img_per_row, i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i//img_per_row, i%img_per_row])
            axes[i//img_per_row, i%img_per_row].axis('off')
            
    plt.show()

def show_h5_dataset_name(ds_path, class_name=None):
    '''
    This is a utility function used to show the dataset names in a hdf5 file.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)
    '''

    with h5py.File(ds_path) as hf:
        if class_name:
            print(hf[class_name].keys())            
        else:
            print(hf.keys())
            
def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):

    '''
    This is a utility function used to load plume images from hdf5 file 
    based on the the ds_name after preprocess with process_func.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)

    :param ds_name: dataset name for plume images in hdf5 file
    :type ds_name: str

    :param process_func: preprocess function
    :type process_func: function(, optional)

    :param show: show the plumes images if show=True
    :type show: bool(, optional)

    '''

    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])
    if show:
        if process_func:
            images = process_func(plumes)
        show_images(images, colorbar=True)
    return plumes
