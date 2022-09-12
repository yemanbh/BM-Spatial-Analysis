# -*- coding: utf-8 -*-
"""
Created on 15/06/2020

@author: yhagos
"""
import os
from skimage import io
from skimage import measure
from skimage.morphology import disk, dilation, erosion
from matplotlib.pyplot import *
from MoSaicNet_Pipeline.configs import seg_coding
from MoSaicNet_Pipeline.utils import seg_data_2_rgb
import multiprocessing as mp

def post_process(slide,
                 input_dir,
				 output_dir_rgb,
				 output_dir_raw,
				 output_dir_binary,
				 area_threshold=5000,
				 classifier_patch_size=40,
                 correct_order = (3, 0, 1, 2),
				 r=6,
				 tissue_id=3,
				 file_name_list=None):
	
	save_dir_raw = os.path.join(output_dir_raw, slide)
	save_dir_rgb = os.path.join(output_dir_rgb, slide)
	save_dir_binary = os.path.join(output_dir_binary, slide)
	os.makedirs(save_dir_raw, exist_ok=True)
	os.makedirs(save_dir_rgb, exist_ok=True)
	os.makedirs(save_dir_binary, exist_ok=True)
	# print(file_name_list)
	for file_name in file_name_list:
		print(f"slide:{slide}; file name:{file_name}")
		output_file_name_rgb = os.path.join(save_dir_rgb, file_name)
		output_file_name_raw = os.path.join(save_dir_raw, file_name)
		output_file_name_binary = os.path.join(save_dir_binary, file_name)
		if all([os.path.isfile(output_file_name_raw), os.path.isfile(output_file_name_binary), os.path.isfile(output_file_name_rgb)]):
			print(f"file already exists; {output_file_name_raw}")
			continue
		im_in = io.imread(os.path.join(input_dir, slide, file_name))
		
		# all pixels labeled as the last class
		im_corrected = correct_order[-1] * np.ones(im_in.shape, dtype='uint8')
		for seg_id in correct_order:
			im_i = im_in == seg_id
			# remove noisy predictions; small areas
			im_i = dilation(im_i, disk(r))
			# morphology analysis
			im_l = measure.label(im_i)
			if len(np.unique(im_l)) != 1:
				props = measure.regionprops_table(im_l, properties=['label', 'area'])
				large_objs = props['label'][props['area'] > area_threshold]
				im_l = (np.isin(im_l, large_objs)) * 1
		
				# erosion
				im_l = erosion(im_l, disk(r))
				# im = np.multiply(im, im_in)
			else:
				# print('Tile without tissue')
				continue
			# fill small holes
			im_invert = 1 - im_l
			im_invert = measure.label(im_invert)
			if len(np.unique(im_invert)) != 1:
				props = measure.regionprops_table(im_invert, properties=['label', 'area'])
				small_holes_id = props['label'][props['area'] < 2 * (classifier_patch_size**2)]
				for k in small_holes_id:
					im_l[im_invert == k] = 1
			# update segmentation result
			im_corrected[np.logical_and(im_l == 1, im_corrected == correct_order[-1])] = seg_id
		# save corrected image in raw format
		io.imsave(output_file_name_raw, im_corrected)
		# save corrected binary image
		im_corrected_binary = 255 * (im_corrected == tissue_id).astype('uint8')
		io.imsave(output_file_name_binary, im_corrected_binary)
		# save corrected image in rgb format
		im_corrected = seg_data_2_rgb(im_corrected, seg_coding)
		io.imsave(output_file_name_rgb, im_corrected)
		
		# Visualization
		# fig, ax = subplots(1, 2)
		# ax[0].imshow(im_in)
		# # sns.scatterplot(x='X', y='Y', data=df, ax=ax[0])
		# ax[1].imshow(im_corrected)
		# show()
		# mask = 255 * (im).astype('uint8')
		
		# io.imsave(output_file_name, mask)
def run(input_dir,
		 output_dir_rgb,
		 output_dir_raw,
		 output_dir_binary,
		slide,
		 area_threshold=5000,
		 classifier_patch_size=40,
         correct_order = (3, 0, 1, 2),
		 r=6, tissue_id=3,
        num_processes=1):
	
	# for slide in os.listdir(input_dir):
	img_files_names_list = os.listdir(os.path.join(input_dir, slide))
	# print(img_files_names_list)
	n = len(img_files_names_list)
	
	if n < num_processes:
		k = n
	else:
		k = num_processes
	
	num_elem_per_process = int(np.ceil(n / k))
	
	file_names_list_list = []
	
	for i in range(k):
		start_ = i * num_elem_per_process
		file_names_list_list.append(img_files_names_list[start_: start_ + num_elem_per_process])
	# print(file_names_list_list)
	# print(file_names_list_list[0])
	# print(file_names_list_list[1])
	# create list of processes

	processes = [
		mp.Process(target=post_process,
									args=(slide,
									      input_dir,
									      output_dir_rgb,
									      output_dir_raw,
									      output_dir_binary,
									      area_threshold,
									      classifier_patch_size,
									      correct_order,
									      r,
									      tissue_id,
									      file_names_list_list[process_num])) for process_num in range(k)]
	
	print('{} processes created'.format(k))
	
	# Run processes
	for p in processes:
		p.start()
	
	# Exit the completed processes
	for p in processes:
		p.join()
	print('All Processes finished!!!')
	
if __name__ == '__main__':
	output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\Resutls\panel1\data\cd_cc_segmentation'
	num_cpu = 12
	for slide_name in os.listdir(os.path.join(output_dir, 'MoSaicNet_Pipeline')):
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
