import json # For dealing with metadata
import os # For file level operations
import time # For timing demonstrations
import datetime # To demonstrate conversion between date and time formats
import glob
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import seaborn as sns

class PlumeMetrics():
    
    '''
    This is a class used to evaluate plume with metrics such as area and velocity.

    :param plumes: input plume images (plume_index*image_index*H*W)
    :type plumes: np.array


    :param condition: PLD growth condition
    :type condition: str

    '''

    
    def __init__(self, plumes, condition):
        self.plumes = plumes
        self.condition = condition
        
        self.metrics_name = ['area', 'area_filled', 'axis_major_length', 
                'axis_minor_length', 'centroid-1', 'centroid-2', 'orientation', 
                'eccentricity', 'perimeter', 'velocity'] 
    
    def crop_clip_image(self, image, x_start=50, intensity=200):
        '''
        This is a function used to crop the image and sharpe the image in naive way. 
        The processed image will be cropped horizontally start from "x_start", and all value to be 0 or given "intensity".

        :param image: input image
        :type image: np.array

        :param x_start: starting x coordinate
        :type x_start: int

        :param intensity: input variable used to determine user want to open which version of digital form, defaults to "parameter" 
        :type intensity: int
        '''

        img_show = np.copy(image)[:,x_start:]
        img_show[img_show<intensity]=0
        img_show[img_show>intensity]=intensity
        return img_show



    def get_metrics(self):

        '''
        This is a function used to calculate metrics based on the plume images.

        '''
        
#         :param show: show the plumes images if show=True
#         :type show: bool(, optional)
                
        metrics_name = ['area', 'area_filled', 'axis_major_length', 
                        'axis_minor_length', 'centroid-0', 'centroid-1', 'orientation', 
                        'eccentricity', 'perimeter']    
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        for i, images in enumerate(self.plumes):
            for m in metrics_name:
                metrics[m].append([])

            for img in images:
                img_show = self.crop_clip_image(img, x_start=50, intensity=200)
                if np.sum(img_show) == 0:
                    for m in metrics_name:
                        metrics[m][i].append(0)
                else:
                    props = regionprops_table(img_show, properties=([
                            'area', 'area_filled', 'axis_major_length', 
                            'axis_minor_length', 'centroid', 'orientation', 
                            'eccentricity', 'perimeter']))
                    data = pd.DataFrame(props)
                    for m in metrics_name:
                        metrics[m][i].append(data[m][0])

        plots_all = []
        for n in metrics_name:
            y_plot = np.stack(metrics[n])
            plots_all.append(y_plot)
        plots_all = np.stack(plots_all)        
        
        dist, velocity = self.calculate_speed()
        plots_all = np.concatenate((plots_all, dist.reshape(1, dist.shape[0], dist.shape[1]), velocity.reshape(1, velocity.shape[0], velocity.shape[1])))
        return  plots_all

    
    def calculate_speed(self):
        
        '''
        This is a function used to calculate plume velocity.

        '''
        
        dist = []
        for plume in self.plumes:
            start = []
            for i, img in enumerate(plume):
                s = []
                for x in range(img.shape[1]):
                    s.append(np.mean(img[:,x]))
                s = np.array(s)
                target_indices = np.where(s>100)

                if target_indices[0].size > 0:
                    p = np.max(target_indices)
                    start.append(p)
                elif i>0:
                    start.append(start[i-1])
                else:
                    start.append(0)  

            dist.append(np.stack(start))

        dist = np.stack(dist, axis=0)
        velocity = dist[:, 1:] - dist[:, :-1]
        velocity = np.concatenate((dist[:,:1], velocity), axis=1)
        return dist, velocity


    def to_df(self, plots_all):
        
        '''
        This is a function to convert numpy array to pandas.DataFrame, which is used to plot in curves with std shadow.

        :param plots_all: plume images
        :type plots_all: np.array

        '''
        
        metrics_name = ['area', 'area_filled', 'axis_major_length', 
                        'axis_minor_length', 'centroid-1', 'centroid-2', 'orientation', 
                        'eccentricity', 'perimeter', 'distance', 'velocity']
        metric_name_index = np.repeat(metrics_name, plots_all.shape[1]*plots_all.shape[2])
        growth_index = list(np.repeat(np.arange(plots_all.shape[1]), plots_all.shape[2]))*plots_all.shape[0]
        time_index = np.array(list(np.arange(plots_all.shape[2]))*plots_all.shape[1]*plots_all.shape[0])
        condition_list = [self.condition]*len(time_index)

        data = np.stack((condition_list, metric_name_index, growth_index, 
                         time_index, plots_all.reshape(-1)))

        df = pd.DataFrame( data=data.T, 
                           columns=['condition', 'metric', 'growth_index', 
                                    'time_step', 'a.u.'] )

        df['growth_index'] = df['growth_index'].astype(np.int32)
        df['time_step'] = df['time_step'].astype(np.int32)
        df['a.u.'] = df['a.u.'].astype(np.float32)
        return df



    
    
def plot_metrics(df, sort_by='condition'):
    
    '''
    This is a function to plot curves based calculated plume metrics.

    :param sort_by: to specify how the plume metrics be sorted. Options: "growth_index" uses plume index as label, "condition" uses different growth condition as label
    :type sort_by: str(, optional)
    
    '''
    
    metrics_name = ['area', 'area_filled', 'axis_major_length', 
                    'axis_minor_length', 'centroid-1', 'centroid-2', 'orientation', 
                    'eccentricity', 'perimeter', 'distance', 'velocity'] 
    
    for metric in metrics_name:
        print(metric)
        sns.set(rc={'figure.figsize':(12,8)})
        sns.set_style("white")

        # bin to 10 growth_index classes
        if sort_by == 'growth_index': 
            df = df.copy()
            start_index_list = np.arange(np.min(df['growth_index']), np.max(df['growth_index']), np.max(df['growth_index'])//10)

            for i in range(len(start_index_list)):
                if i == len(start_index_list)-1:
                    for index in range(start_index_list[i], np.max(df['growth_index'])):
                        df['growth_index'] = df['growth_index'].replace(index, start_index_list[i])
                else:
                    for index in range(start_index_list[i], start_index_list[i+1]):
                        df['growth_index'] = df['growth_index'].replace(index, start_index_list[i])

        plot = sns.lineplot(data=df[df['metric']==metric], 
                            x='time_step', y='a.u.', hue=sort_by)
        plt.show()
    return df



def plot_metrics_heatmap(df, frame_range):
    
    '''
    This is a function to plot heatmap based calculated plume metrics.

    :param frame_range: to specify the heatmap start from which frame  
    :type frame_range: tuple(int, int)
    
    '''
    
    for m in df.metric.unique():
        df_hm = df.loc[df['metric']==m]
        df_hm = df_hm.loc[df_hm['time_step']<frame_range[1]]
        df_hm = df_hm.loc[df_hm['time_step']>frame_range[0]]
        df_hm = df_hm.pivot("growth_index", "time_step", "a.u.")
        ax = sns.heatmap(df_hm).set(title=m)
        plt.show()