import shutil
import sys
import json
import datetime
import os, glob, h5py
import numpy as np
import matplotlib.pyplot as plt
from datafed.CommandLib import API

def remove_desktop_ini(path):

    '''
    This function is designed to remove "desktop.ini" file in 4 sub-directories under given path 

    :param path: the path of target directory
    :type path: str

    '''

    if path[-1] == '/': path = path[:-1]
        
    # level 1
    if os.path.isdir(path):
        for folder_1 in glob.glob(path+'/*'):
            if folder_1.split('/')[-1].split('\\')[-1] == 'desktop.ini':
                os.remove(folder_1)  

            # level 2
            if os.path.isdir(folder_1):
                for folder_2 in glob.glob(folder_1+'/*'):
                    if folder_2.split('/')[-1].split('\\')[-1] == 'desktop.ini':
                        os.remove(folder_2)  

                    # level 3
                    if os.path.isdir(folder_2):
                        for folder_3 in glob.glob(folder_2+'/*'):
                            if folder_3.split('/')[-1].split('\\')[-1] == 'desktop.ini':
                                os.remove(folder_3)  

                            # level 4
                            if os.path.isdir(folder_3):
                                for folder_4 in glob.glob(folder_3+'/*'):
                                    if folder_4.split('/')[-1].split('\\')[-1] == 'desktop.ini':
                                        os.remove(folder_4) 
                                        
                                        
def pack_to_hdf5_and_upload(file_path, file_name, growth_para):

    '''
    This is a function used to pack a folder of PLD plume 
    and upload to DataFed server with recorded growth condition

    :param file_path: the path of PLD plume directory
    :type file_path: str

    :param file_name: the name of the hdf5 file
    :type file_name: str

    :param growth_para: recorded pld growth condition
    :type growth_para: dict

    '''

    pack_to_hdf5(file_path, file_name)
    upload_to_datafed(file_path, file_name, growth_para, dataset_id='c/391937642')


def pack_to_hdf5(file_path, file_name):

    '''
    This function will read the images in folders(plumes) under "ds_path/BMP/<target_name>" 
    and convert them into a hdf5 file with following data struction:

    file_name:ds_path.h5
      group: PLD_Plumes
        dataset: target_name(SrRuO3) = n_videos*n_frames*H*W (np.float32)
    

    :param file_path: the path of PLD plume directory
    :type file_path: str

    :param file_name: the name of the hdf5 file
    :type file_name: str

    '''

    print('Packing to hdf5...')
    ds_path = file_path + '/' + file_name

#     growth_para = self.get_info()   

    # make sure the format is without '/'
    if ds_path[-1] == '/': ds_path[:-1]

    # overwrite the original .h5 file
    if os.path.isfile(ds_path + '.h5'): os.remove(ds_path + '.h5')
        

    with h5py.File(ds_path + '.h5', mode='a') as h5_file:
        h5_group_plume = h5_file.create_group('PLD_Plumes')

        # remove desktop.ini file from all sub-directory
        remove_desktop_ini(ds_path)

        for target_folder in os.listdir(ds_path+'/'):
            if target_folder == '.ipynb_checkpoints': continue 

            length_list = []
            target_path = ds_path + '/' + target_folder

            if os.listdir(target_path) == []:
                os.rmdir(target_path)                
            
            # define the image shape
            for plume_folder in os.listdir(target_path + '/BMP'):
                length_list.append(len(os.listdir(target_path + '/BMP/' + plume_folder)))
                img_shape = plt.imread(glob.glob(target_path+'/BMP/'+plume_folder+'/*')[0]).shape

            # assign to hdf5 dataset
            create_data = h5_group_plume.create_dataset(target_folder, dtype=np.uint8, 
                                                        shape=((len(length_list), max(length_list), 
                                                                img_shape[0], img_shape[1]))) # dtype=np.uint8
            for i, plume_folder in enumerate(os.listdir(target_path + '/BMP')):
                for j, file in enumerate(glob.glob(target_path + '/BMP/' + plume_folder + '/*')):
                    create_data[i, j] = plt.imread(file)

    print('Done!')


def upload_to_datafed(file_path, file_name, growth_para, dataset_id='c/391937642'):

    '''
    This function will upload a hdf5 file to DataFed server with specified "dataset_id"
    An Globus Endpoint is required to set up in local server before use this funciton  

    :param file_path: the path of PLD plume directory
    :type file_path: str

    :param file_name: the name of the hdf5 file
    :type file_name: str

    :param growth_para: recorded pld growth condition
    :type growth_para: dict

    :param dataset_id: DataFed dataset id for uploading
    :type dataset_id: str

    '''


    print('Uploading to Datafed...')

    df_api = API()
    dc_resp = df_api.dataCreate(file_name,
                                metadata=json.dumps(growth_para),
                                parent_id=dataset_id, # parent collection
                               )
    rec_id = dc_resp[0].data[0].id
    put_resp = df_api.dataPut(rec_id, # record id
                              file_path+'/'+file_name+'.h5', # path to file
                              wait=True  # Waitcas until transfer completes.
                              )
    out = put_resp
    print('Done!')
    return out