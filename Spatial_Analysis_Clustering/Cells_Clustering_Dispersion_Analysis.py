import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pointpats import PointPattern
import numpy as np

def get_knn_z_value(input_points, tissue_area):
	# compute z value
	n = len(df)
	# observed mean distance
	pp = PointPattern(input_points)
	mean_nnd = pp.mean_nnd
	# expected  mean distance for features given in a random pattern
	mean_nnd_random = 0.5 / (np.sqrt(n / tissue_area))
	# standard error
	se = 0.26136 / (np.sqrt(n ** 2 / tissue_area))
	
	z = (mean_nnd - mean_nnd_random) / se
	
	return z

def get_nnd_stat(csrp_dist, d_mean_obs):
	z = (d_mean_obs - np.mean(csrp_dist)) / np.std(csrp_dist)
	# p value
	if d_mean_obs < np.mean(csrp_dist):
		p = np.sum(csrp_dist < d_mean_obs) / len(csrp_dist)
	else:
		p = np.sum(csrp_dist > d_mean_obs) / len(csrp_dist)
	
	return np.mean(csrp_dist), z, p


if __name__ == '__main__':

	seg_image_ext = '.npy'
	mpp = 0.4606
	scale = 16
	tissue_label = 3
	num_rand_samples = 300

	knn_k_values = [1]

	#placeholder
	nn_data = pd.DataFrame(columns=['SlideName', 'CellName', '1nn_z_value',
									'1nn_p_value'] + [f'{k}nn_dist (um)' for k in knn_k_values])

	# for panel in ['panel1', 'panel2']:

	cell_pos_dir = r'XX/YY/Combined_csv.csv'
	mosaicnet_data_dir = r'XX/YY/MoSaicNet/Output'
	output_dir = r'path/to/output_dir'


	cell_name = ['FOXP3+CD4+', 'CD8+', 'FOXP3-CD4+']
	os.makedirs(output_dir, exist_ok=True)
	file_names = [file_name for file_name in os.listdir(mosaicnet_data_dir) if file_name.endswith(seg_image_ext)]

	for i, file_name in enumerate(file_names):

		slide_name = file_name.replace(seg_image_ext, '')
		# if slide_name != 'CD4_BLIMP1_CD8_UH15-19506.ndpi':
		# 	continue
		seg_image = np.load(os.path.join(mosaicnet_data_dir, file_name))
		tissue_region = 1 * (seg_image == tissue_label)
		tissue_area = np.sum(seg_image == tissue_label)

		cell_pos = pd.read_csv(os.path.join(cell_pos_dir, slide_name + '.csv'))

		df = cell_pos.loc[cell_pos['Class'] == cell_name, ['X', 'Y', 'Class']]

		# tissue region coordinates
		row_tissue, col_tissue = np.where(tissue_region)
		m = len(row_tissue)
		tissue_pos = np.hstack((np.reshape(col_tissue, (m, 1)), np.reshape(row_tissue, (m, 1))))



		# ss1 scaling
		df[['X', 'Y']] = df[['X', 'Y']] / scale

		# VISUALIZATION sample
		# plt.imshow(seg_image, interpolation='nearest', cmap='Set2')
		# sns.scatterplot(x='X', y='Y', data=df, hue='Class', s=.7, edgecolor=None)
		# plt.show()
		# change to microns
		tissue_pos = tissue_pos * mpp
		df[['X', 'Y']] = df[['X', 'Y']] * mpp

		points = df.to_numpy()
		# knn_dist_values = []
		# for k_value in knn_k_values:
		pp = PointPattern(points)
		knn = pp.knn(k=1)
		# mean_knn_dist = np.mean(np.max(knn[1], axis=1))
		# knn_dist_values.append(np.mean(np.max(knn[1], axis=1)))
		nnd_mean_observed = np.mean(knn[1])
		rand_process_nnd = []
		for _ in range(num_rand_samples):
			csrp = tissue_pos[np.random.choice(tissue_pos.shape[0], len(df), replace=False), :]

			# plt.imshow(seg_image == 3, interpolation='nearest', cmap='Set2')
			# sns.scatterplot(x=csrp[:, 0] / mpp, y=csrp[:, 1] / mpp, s=.7, edgecolor=None)
			# plt.show()

			pp = PointPattern(csrp)
			knn = pp.knn(k=1)
			rand_process_nnd.append(np.mean(knn[1]))

		# z = get_knn_z_value(points, tissue_area)
		nnd_mean_random, z, p = get_nnd_stat(rand_process_nnd, nnd_mean_observed)
		print(nnd_mean_observed, nnd_mean_random, z, p)
		nn_data.loc[nn_data.__len__()] = [slide_name, cell_name, z,  p, nnd_mean_observed]

		nn_data.to_csv(os.path.join(output_dir, f"dispersion_nn_data.csv"), index=False)



