# -*- coding: utf-8 -*-
"""
Created on 17/06/2020

@author: yhagos
"""

import os
from AwareNet_Annotation.utils.image_annotate_cells import map_cell_2_colour, run
# from parse_arguments import get_parsed_arguments

def annotate_results(csv_dir, cws_dir, output_dir, num_processes, slide_name, cell_label_text):
	
	# args = get_parsed_arguments()
	
	# if args.cluster is True:
	#
	# 	num_processes = int(args.num_cpu)
	# 	multi_process = args.mp
	# 	csv_dir = args.csv_dir
	# 	cws_dir = args.cws_dir
	# 	src_dir = args.src_dir
	# 	output_dir = args.output_dir
	# 	batch = int(args.batch)
	# else:
	# 	num_processes = 10
	# 	multi_process = True
	# 	batch = -1
	#
	# 	csv_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\Resutls\20201118_CD_CC_Panel1\results\AnnotatedCellsCoord_fp_removed'
	# 	cws_dir =  r'Y:\yhagos\Myeloma_BM_mapping\raw-data\20201026_diagnostic_panel1\cws'
	# 	output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\Resutls\20201118_CD_CC_Panel1\results\AnnotatedTiles_fp_removed'
	# 	src_dir = os.getcwd()
		
	# cell_label_text = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'cell_label_panel2.txt')
	
	color_to_cell_dict = map_cell_2_colour(cell_label_text)
	
	params = dict(output_dir=output_dir,
				  images_dir=cws_dir,
				  csv_dir=csv_dir,
				  image_ext=['.jpg'],
				  color_to_cell_dict=color_to_cell_dict,
				  num_processes=num_processes,
				  slide_name=slide_name)
	
	run(**params)

	print('DONE!')
