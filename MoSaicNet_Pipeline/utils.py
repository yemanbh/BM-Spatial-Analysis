# -*- coding: utf-8 -*-
"""
Created on 01/02/2021

@author: yhagos
"""
import os
import numpy as np
import pandas as pd
from skimage import io

def map_seg_val_2_rgb(channel_coding):
    seg_to_rgb = dict()
    for key, hex_val in channel_coding.items():
        # the second split if the there is a comment without space
        val = hex_val.replace('#', '')
        # change color to rgb
        seg_to_rgb[key] = tuple(int(val[i:i + 2], 16) for i in (0, 2, 4))
        # color_to_cell_dict[cell_name] = color_val

    return seg_to_rgb


def seg_data_2_rgb(im, seg_coding):
    seg_im = np.repeat(np.expand_dims(im, axis=-1), 3, axis=-1)
    seg_val_2_color = map_seg_val_2_rgb(seg_coding)
    for v, rgb in seg_val_2_color.items():
        mask = np.all(seg_im == [v] * 3, axis=-1)
        seg_im[mask, :] = rgb
        
    return seg_im


def get_tissue_area_in_micros(input_dir, region_id_dict, mpp):
    region_names = list(region_id_dict.keys())
    col = ['SlideName']+ region_names
    area_df = pd.DataFrame(columns=col)
    # get area in pixels ^2
    
    for slide in os.listdir(input_dir):
        print(slide)
        area = dict()
        for r_name in region_names:
            area[r_name] = 0
            
        for file_name in os.listdir(os.path.join(input_dir, slide)):
            print(file_name)
            im = io.imread(os.path.join(input_dir, slide, file_name))
            for r_name in region_names:
                area[r_name] = area[r_name] + np.sum(im == region_id_dict[r_name])
            # tissue_area += np.sum(im == tissue_id)
        # update record
        record = area
        record['SlideName'] = slide
        area_df = area_df.append(record, ignore_index=True)
        # area_df.loc[len(area_df)] = [slide, tissue_area]
    # pixel to micros
    area_df[region_names] = area_df[region_names].astype('float64') * (mpp ** 2)
    
    # get block number
    # area_df['Block no.'] = area_df['file_name'].map(lambda v: '-'.join(v.split('-')[:2]))
    
    return area_df
