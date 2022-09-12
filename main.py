# -*- coding: utf-8 -*-
"""
Created on 29/03/2021

@author: yhagos
"""

# basic libraries
import os
from time import time

# arguemnts
from parse_arguments import get_parsed_arguments


# Configuration file
from Configuration import my_configuration


# user defined modules
from MoSaicNet_Pipeline.run_tissue_seg import run_tissue_seg
from AwareNet_pipeline.run_cd_cc import run_cd_cc
from Remove_FP_Detection.remove_fp_pred import remove_fp_predictions
from AwareNet_Annotation.draw_gt_markers import annotate_results
from Combine_CSV.CombineCellAnnotationCSV import combine_cws_csv
from Cell_Counting.count_cells import count_cells_cws_combined

if my_configuration.running_mode == 'terminal':
    # get arguments from Configuration file
    output_dir = my_configuration.output_dir
    data_dir = my_configuration.data_dir
    num_cpu = my_configuration.num_cpu
elif my_configuration.running_mode == 'remote':
    # get arguments from job submission file
    args = get_parsed_arguments()

    output_dir = args.output
    data_dir = args.data_dir
    num_cpu = int(args.num_cpu)
else:
    raise Exception('unknown value for param running_mode. Allowed values are remote or terminal, but given {}'.format(my_configuration.running_mode))


cws_dir = os.path.dirname(data_dir)
slide_name = os.path.basename(data_dir)

# record time take per slide
start_time = time()


# tissue segmentation
print("************************** Tissue segmentation **************************")
run_tissue_seg(output_dir=os.path.join(output_dir, 'tissue_seg'),
               cws_dir=cws_dir,
               slide_name=slide_name,
               num_cpu=num_cpu)

print("************************** Cell detection and classification **************************")
run_cd_cc(cws_dir,
          output_dir=os.path.join(output_dir, 'AwareNet_pipeline'),
          cws_mask=os.path.join(output_dir, 'tissue_seg', 'superpixel_seg_binary'),
          slide_name=slide_name,
          cell_names_ordered=my_configuration.cell_names_ordered,
          cell_label_text=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Configuration', 'cell_annotation_labels.txt'),
          num_cpu=num_cpu,
          pred_prob_threshold=my_configuration.pred_prob_threshold,
          area_threshold=my_configuration.area_threshold)

# remove false positives
print("************************** Remove FP detection **************************")
remove_fp_predictions(csv_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'AnnotatedCellsCoord'),
                      seg_dir=os.path.join(output_dir, 'tissue_seg', 'superpixel_seg_binary'),
                      output_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'AnnotatedCellsCoord_fp_removed'),
                      slide=slide_name)

# annotate data
print("************************** Annotation **************************")
annotate_results(csv_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'AnnotatedCellsCoord_fp_removed'),
                 cws_dir=cws_dir,
                 cell_label_text=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Configuration', 'cell_annotation_labels.txt'),
                 output_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'annotatedTiles')
                 , num_processes=num_cpu, slide_name=slide_name)


# combine csv data
print("************************** Combine csv detection **************************")
combine_cws_csv(csv_files_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'AnnotatedCellsCoord_fp_removed'),
                cws_dir=cws_dir,
                slide_name=slide_name,
                output_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'Combined_csv_fp_removed'),
                down_sample_factor=1
                )

# combine csv data
print("************************** Combine csv detection **************************")

count_cells_cws_combined(input_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'Combined_csv_fp_removed'),
                         output_dir=os.path.join(output_dir, 'AwareNet_pipeline', 'Cell_count'),
                         class_col_name='Class',
                         cell_names=list(set(my_configuration.cell_names_ordered)))

end_time = time()

os.makedirs(os.path.join(output_dir, 'time_elapsed'), exist_ok=True)
with open(os.path.join(output_dir, 'time_elapsed', slide_name + '.txt'), 'w') as txt:
    txt.write(f"{(end_time - start_time) / 60} minutes")

