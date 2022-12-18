import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.measure import label, regionprops_table
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, dilation, medial_axis
import numpy as np





mosaicnet_data_dir = r'sample_input'
output_dir = r'sample_output'

# segmentation data output image extension
seg_image_ext = '.npy'

# microns per pixel of the images
mpp = 0.4606

# scale of the segmentation image
scale = 16

# bone lagel
bone_label = 1


os.makedirs(output_dir, exist_ok=True)
file_names = [file_name for file_name in os.listdir(mosaicnet_data_dir) if file_name.endswith(seg_image_ext)]

# as dict
bone_thickness_dict = dict()
# as table
bone_thickness_df = pd.DataFrame(columns= ['SlideName', 'Mean_Max_thickness', 'Mean_Mean_thickness'])

for k, file_name in enumerate(file_names):
	slide_name = file_name.replace(seg_image_ext, '')
	print(f'processing {slide_name}')
	# get regions
	seg_image = np.load(os.path.join(mosaicnet_data_dir, file_name))
	bones = 1 * (seg_image == bone_label)
	
	# preprocessing
	bones_labeled = label(bones)
	props_df = pd.DataFrame(regionprops_table(bones_labeled, properties=('label', 'area')))

	artifact_area_threshold = 400
	bones_props = props_df.loc[props_df['area'] > artifact_area_threshold, :]
	correct_bones = np.isin(bones_labeled, test_elements=bones_props['label'].to_list())
	
	correct_bones_final = dilation(correct_bones, selem=disk(1))
	correct_bones_final = binary_fill_holes(correct_bones_final)
	
	labeled_img = label(correct_bones_final)
	bones_ids = [l for l in np.unique(labeled_img) if l != 0]
	
	max_thickness_values = []
	mean_thickness_values = []
	for bone_id in bones_ids:
		selected_bone = 1 * (labeled_img == bone_id)
		
		# This section is skeletonization
		# Compute the medial axis (skeleton) and the distance transform
		skel, distance = medial_axis(selected_bone.copy(), return_distance=True)
		
		# Distance to the background for pixels of the skeleton
		dist_on_skel = distance[skel]
		
		# thickness as double of distance from background to skeleton
		thickness_values = 2 * dist_on_skel
		
		# change unit from ss1 pixels to microns
		thickness_values = thickness_values * mpp * scale
		max_thickness_values.append(np.max(thickness_values))
		
		# take top 80 percent
		q_20 = np.quantile(thickness_values, q=0.2)
		
		mean_thickness_values.append(np.mean(thickness_values[thickness_values > q_20]))
		
	
	bone_thickness_dict[slide_name] = dict(mean_thickness = np.mean(mean_thickness_values), max_thickness = np.max(mean_thickness_values))
	bone_thickness_df.loc[len(bone_thickness_df)] = [slide_name, np.mean(max_thickness_values), np.mean(mean_thickness_values)]

# save data
bone_thickness_df.to_csv(os.path.join(output_dir, 'bone_thickness_csv.csv'), index=False)
with open(os.path.join(output_dir, 'bone_thickness_dict.json'), 'w') as j_fp:
	json.dump(bone_thickness_dict, fp=j_fp, indent=3)


	



		
		
	
