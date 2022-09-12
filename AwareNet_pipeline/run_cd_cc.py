# -*- coding: utf-8 -*-
"""
Created on 14/04/2020

@author: yhagos
"""
import os
import json

from AwareNet_pipeline.CellSpatialMapping import DetectCells

def run_cd_cc(cws_dir,
			  output_dir,
			  slide_name,
			  cell_label_text,
			  cell_names_ordered,
			  cws_mask=None,
			  img_name_pattern=('Da',),
			  num_cpu=1,
			  pred_prob_threshold=0.5,
			  distance_threshold=10,
			  area_threshold=3):
	cc_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Best_models', 'CC_best_model', 'best_model.h5')
	cd_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Best_models', 'CD_best_model', 'best_model.h5')


	params = dict(
		cell_detection_model_dir=cd_model_dir,
		cell_classification_model_dir=cc_model_dir,
		input_dir=cws_dir,
		output_dir=output_dir,
		num_processes=num_cpu,
		slide_name=slide_name,
		scale=1,
		run_on_batch=True,
		split_area_threshold=0.95,
		pred_probability_threshold=pred_prob_threshold,
		postprocess=True,
		cell_names_ordered=cell_names_ordered,     # ['Blue', 'Brown', 'Red', 'RedBrown']
		cell_label_text=cell_label_text,
		do_classification=True,
		prediction_prob_area_threshold=area_threshold,
		distance_threshold=distance_threshold,
		count_loss_used=False,
		img_name_pattern=img_name_pattern,
		cws_mask=cws_mask,
		save_annotated_image=True,
		save_detection_prob_map=False,
	)
	
	# save hyper parameters
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, '_params_.json'), 'w') as fp:
		json.dump(params, fp, indent=5)
	
	# run
	obj = DetectCells(**params)
	obj.run()
