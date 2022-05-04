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


def process_func(images):
    images = images[np.random.randint(0, images.shape[0])]
    return images

def show_images(images, labels=None, img_per_row=8, colorbar=False):
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

def show_h5_dataset_name(ds_path, class_name):
    with h5py.File(ds_path) as hf:
        if class_name:
            print(hf[class_name].keys())            
        else:
            print(hf.keys())
            
def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):
    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])
    if show:
        if process_func:
            images = process_func(plumes)
        show_images(images, colorbar=True)
    return plumes

def get_metrics(plumes, image_process, show=True):
    
    metrics_name = ['area', 'area_filled', 'axis_major_length', 
                    'axis_minor_length', 'centroid-0', 'centroid-1', 'orientation', 
                    'eccentricity', 'perimeter']    
    metrics = {}
    for m in metrics_name:
        metrics[m] = []

    for i, images in enumerate(plumes):
        for m in metrics_name:
            metrics[m].append([])

        for img in images:
            img_show = image_process(img)
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
                    
    plots_mean = []
    plots_all = []
    for n in metrics_name:
        y_plot = np.stack(metrics[n])
        plots_all.append(y_plot)
        plots_mean.append(np.mean(y_plot, axis=0))
        if show:
            h = plt.plot(np.mean(y_plot, axis=0), label=n)  
    if show:
        leg = plt.legend(loc='upper right')
        plt.show()
    
    plots_all = np.stack(plots_all)
    plots_mean = np.stack(plots_mean)
    return plots_mean, plots_all

def process_func(images):
    images = images[np.random.randint(0, images.shape[0])]
    return images

def image_process(img):
    img_show = np.copy(img)[:,50:]
    img_show[img_show<200]=0
    img_show[img_show>200]=200
    return img_show

def calculate_speed(plumes):
    velocity_all = []
    for plume in plumes:
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
                
        velocity_all.append(np.stack(start))
        
    velocity_all = np.stack(velocity_all, axis=0)
    velocity_mean = np.mean(velocity_all, axis=0)
    return velocity_mean, velocity_all

def convert_df(plots_all, condition):
    metrics_name = ['area', 'area_filled', 'axis_major_length', 
                    'axis_minor_length', 'centroid-1', 'centroid-2', 'orientation', 
                    'eccentricity', 'perimeter', 'velocity']  
    metric_name_index = np.repeat(metrics_name, plots_all.shape[1]*plots_all.shape[2])
    growth_index = list(np.repeat(np.arange(plots_all.shape[1]), plots_all.shape[2]))*plots_all.shape[0]
    time_index = np.array(list(np.arange(plots_all.shape[2]))*plots_all.shape[1]*plots_all.shape[0])
    condition_list = [condition]*len(time_index)
    
    data = np.stack((condition_list, metric_name_index, growth_index, 
                     time_index, plots_all.reshape(-1)))

    df = pd.DataFrame( data=data.T, 
                       columns=['condition', 'metric', 'growth_index', 
                                'time_step', 'a.u.'] )

    df['growth_index'] = df['growth_index'].astype(np.int32)
    df['time_step'] = df['time_step'].astype(np.int32)
    df['a.u.'] = df['a.u.'].astype(np.float32)
    return df

def plumes_to_df(ds_path, ds_name, condition, show_plume=False):
    plumes = load_h5_examples(ds_path, 'PLD_Plumes', ds_name, process_func, show=False)
    if show_plume:
        show_images(np.mean(plumes, axis=0))

    plots_mean, plots_all = get_metrics(plumes, image_process, show=False)
    velocity_mean, velocity_all = calculate_speed(plumes)

    plots_mean = np.concatenate((plots_mean, velocity_mean.reshape(1, -1)))
    plots_all = np.concatenate((plots_all, velocity_all.reshape(1, velocity_all.shape[0], velocity_all.shape[1])))

    df = convert_df(plots_all, condition)
    return df


def plot_metrics(df, metrics_name, label_with='condition'):
    for metric in metrics_name:
        print(metric)
        sns.set(rc={'figure.figsize':(12,8)})
        sns.set_style("white")

        # bin to 10 growth_index classes
        if label_with == 'growth_index': 
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
                            x='time_step', y='a.u.', hue=label_with)
        plt.show()
    return df