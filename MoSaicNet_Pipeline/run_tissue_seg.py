# -*- coding: utf-8 -*-
"""
Created on 14/04/2020

@author: yhagos
"""
import os
# import json
from MoSaicNet_Pipeline.S0_superpixel_seg_mp import TissueSegmentation
# from MoSaicNet_Pipeline.parse_arguments import get_parsed_arguments
from MoSaicNet_Pipeline.S1_dl_seg_to_rgb import convert_dl_result_2_rgb
from MoSaicNet_Pipeline.S2_post_processing import run
from MoSaicNet_Pipeline.S3_tissue_area import get_tissue_area

def run_tissue_seg(output_dir, cws_dir, slide_name, num_cpu=1, img_name_pattern=('Da',)):
	# args = get_parsed_arguments()
	# if args.cluster:
	# 	model_dir = args.model_dir
	# 	output_dir = args.output
	# 	cws_dir = args.cws_dir
	# 	num_cpu = int(args.num_cpu)
	# 	multi_process = args.mp
	# 	run_on_batch = args.run_on_batch
	# 	batch_index = int(args.batch)
	# else:
	# 	num_cpu = 4
	# 	multi_process = True
	# 	run_on_batch = True
	# 	batch_index = -1
	# 	output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\panel-1_new'
	# 	cws_dir = r'Y:\yhagos\Myeloma_BM_mapping\raw-data\20201026_diagnostic_panel1\cws\new_panel1'
	# 	model_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\BestModel_40\best_model.h5'
	
	params = dict(model_weight_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg_model', 'best_model.h5'),
	              input_dir=cws_dir,
	              output_dir=os.path.join(output_dir, 'MoSaicNet_Pipeline'),
	              num_processes=num_cpu,
	              img_name_pattern=img_name_pattern,
	              slide_name=slide_name
	              )
	# save hyper parameters
	# os.makedirs(output_dir, exist_ok=True)
	# with open(os.path.join(output_dir, '_params_.json'), 'w') as fp:
	# 	json.dump(params, fp, indent=5)
	
	# dl segmentation
	obj = TissueSegmentation(**params)
	obj.run()
	
	# to rgb
	convert_dl_result_2_rgb(seg_dir=os.path.join(output_dir, 'MoSaicNet_Pipeline'),
	                        output_dir=os.path.join(output_dir, 'superpixel_seg_rgb'),
	                        slide=slide_name)
	
	# post processing
	run(input_dir=os.path.join(output_dir, 'MoSaicNet_Pipeline'),
	    output_dir_rgb=os.path.join(output_dir, 'superpixel_seg_rgb_corrected'),
	    output_dir_raw=os.path.join(output_dir, 'superpixel_seg_raw_corrected'),
	    output_dir_binary=os.path.join(output_dir, 'superpixel_seg_binary'),
	    slide=slide_name,
	    area_threshold=5000,
	    classifier_patch_size=40,
	    correct_order=(3, 0, 1, 2),
	    r=6,
	    num_processes=num_cpu)
	
	# tissue area
	get_tissue_area(input_dir=os.path.join(output_dir, 'superpixel_seg_raw_corrected'),
	                output_dir=os.path.join(output_dir, 'tissue_area'))
		# output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\tissue_area'
		# params = dict(
		# 	input_dir=r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\superpixel_seg_raw_corrected',