from skimage import io
import os
import numpy as np
from matplotlib.pyplot import *
from MoSaicNet_Pipeline.configs import seg_coding
from MoSaicNet_Pipeline.utils import map_seg_val_2_rgb


# seg_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\panel-1_new\MoSaicNet_Pipeline'
# output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\Result\2021-01-31_superpixel_tissue_seg\data\panel-1_new\superpixel_seg_rgb'

def convert_dl_result_2_rgb(seg_dir, output_dir, slide):
    save_dir = os.path.join(output_dir, slide)
    os.makedirs(save_dir, exist_ok=True)
    for file_name in os.listdir(os.path.join(seg_dir, slide)):
        # cws_im = io.imread(os.path.join(cws_dir, file_name))
        seg_im = io.imread(os.path.join(seg_dir, slide, file_name))
        seg_im = np.repeat(np.expand_dims(seg_im, axis=-1), 3, axis=-1)
        seg_val_2_color = map_seg_val_2_rgb(seg_coding)
        for v, rgb in seg_val_2_color.items():
            mask = np.all(seg_im == [v]*3, axis=-1)
            seg_im[mask, :] = rgb
        io.imsave(os.path.join(save_dir, file_name), seg_im)


# fig, ax = subplots(1, 2)
# ax[0].imshow(cws_im)
# ax[0].axis('off')
# ax[1].imshow(seg_im, cmap='gray')
# # fig.colorbar(img)
# ax[1].axis('off')
#
# show()