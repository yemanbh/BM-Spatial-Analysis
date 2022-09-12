# -*- coding: utf-8 -*-
"""
Created on 05/02/2021

@author: yhagos
"""
import os
from MoSaicNet_Pipeline.utils import get_tissue_area_in_micros
region_id_dict = {
    'tissue': 3,
    'fat': 2,
    'bone': 1,
    'blood': 0
    }
def get_tissue_area(output_dir, input_dir):
    # output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\tissue_area'
    # r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\superpixel_seg_raw_corrected',
    params = dict(input_dir=input_dir,
                  region_id_dict=region_id_dict,
                  mpp=0.4606)
    df = get_tissue_area_in_micros(**params)
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'region_areas_in_mpp_.csv'), index=False)
