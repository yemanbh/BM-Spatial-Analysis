# -*- coding: utf-8 -*-
"""
Created on 01/02/2021

@author: yhagos
"""
import os
import pandas as pd
from skimage import io
import numpy as np
from matplotlib.pyplot import *
import seaborn as sns


def remove_fp_predictions(csv_dir, seg_dir, output_dir, slide):
   
    save_dir = os.path.join(output_dir, slide)
    os.makedirs(save_dir, exist_ok=True)
    for file_name in os.listdir(os.path.join(csv_dir, slide)):
        print(file_name)
        df = pd.read_csv(os.path.join(csv_dir, slide, file_name))
        # print(df.columns)
            # continue
        seg_img = io.imread(os.path.join(seg_dir, slide,
                                         os.path.splitext(file_name)[0] + '.jpg'))
        if len(df) == 0 or np.sum(seg_img == 255) == 0:
            # print('there are no cells in this tile')
            df = pd.DataFrame(columns=['Class', 'X', 'Y'])
        else:
            # tissue area x and y coordinate
            y, x = np.where(seg_img == 255)
            xy_coord_seg = list(zip(list(x), list(y)))
            # cells location x and y coordinate
            xy_loc_cells = df[['X', 'Y']].to_numpy().tolist()
            between_tuple = list(map(tuple, xy_loc_cells))
            # find intercection
            intersection_list = list(set(between_tuple).intersection(map(tuple, xy_coord_seg)))
            is_cell_inside = list(map(lambda coord: coord in intersection_list, between_tuple))
            df['flag'] = is_cell_inside
            df = df.loc[df['flag'] ==True, :]
            if len(df) == 0:
                df = pd.DataFrame(columns=['Class', 'X', 'Y'])
            # visualization
            # fig, ax = subplots(1, 2)
            # im = io.imread(os.path.join(cws_dir, slide, os.path.splitext(file_name)[0] + '.jpg'))
            # ax[0].imshow(seg_img == tissue_id)
            # sns.scatterplot(x='X', y='Y', data=df, ax=ax[0])
            # ax[1].imshow(seg_img==tissue_id)
            # show()
        df.to_csv(os.path.join(save_dir, file_name))
    assert len(os.listdir(os.path.join(csv_dir, slide))) == len(os.listdir(save_dir))