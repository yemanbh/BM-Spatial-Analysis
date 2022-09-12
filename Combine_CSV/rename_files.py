# -*- coding: utf-8 -*-
"""
Created on 03/02/2021

@author: yhagos
"""
import os
input_dir = r'Z:\yhagos\Projects\Myeloma_BM_mapping\Results\20201118_CD_CC_Panel1\results\Combined_csv'
# output_dir = r'Z:\yhagos\Projects\Myeloma_BM_mapping\Results\20201118_CD_CC_Panel1\results\Combined_csv_edited'

# os.makedirs(output_dir, exist_ok=True)
for file_name in os.listdir(input_dir):
    if not file_name.endswith('csv'):
        continue
    file_name_new = '-'.join(file_name.split('-')[:2]) + '.csv'
    
    os.rename(os.path.join(input_dir, file_name), os.path.join(input_dir, file_name_new))
