# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:45:31 2018

@author: yhagos
"""
import os
from skimage import io
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import regionprops_table
import pandas as pd

class TissueSegmentation(object):
    
    def __init__(self, model_weight_dir,
                 input_dir,
                 output_dir,
                 slide_name,
                 num_processes=1,
                 img_name_pattern=('Da'),
                 normalization='regular',
                 stride=None,
                 overlap=False,
                 std_value=0.5,
                 u_value=2000,
                 compactness_value=30
                 ):
        
        self.model_weight_dir = model_weight_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.overlap = overlap
        self.normalization = normalization
        # self.num_classes = num_classes
        self.img_name_pattern = img_name_pattern
        self.stride = stride
        self.u_value = u_value
        self.compactness_value = compactness_value
        self.std_value = std_value
        self.slide_name = slide_name
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def eval_tiles(self, batch_name, file_names_list, p_n):
        
        num_files = len(file_names_list)
        
        # load model and get input shape
        model = tf.keras.models.load_model(filepath=self.model_weight_dir, compile=False)
        patch_size = model.input_shape[1]
            
        input_im_folder = os.path.join(self.input_dir, batch_name)
        output_folder = os.path.join(self.output_dir, batch_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for jj, image_name in enumerate(file_names_list):
            
            print('Batch:{}, process:{}, Tile:{}/{}'.format(batch_name, p_n + 1, jj + 1, num_files))
            'output file names'
            seg_img_full_path = os.path.join(output_folder, image_name)
            file_full_path = os.path.join(input_im_folder, image_name)
            print("test 1:", file_full_path)
            # check if file name exists
            if os.path.isfile(seg_img_full_path) is True:
                print('file already exists')
                continue
            else:
                pass
            im = io.imread(file_full_path)
            
            # padding
            pad_size = 2 * patch_size
            im_pad = np.zeros((2 * pad_size + im.shape[0], 2 * pad_size + im.shape[1], 3), dtype='uint8')
            im_pad[pad_size:im.shape[0] + pad_size, pad_size:im.shape[1] + pad_size, :] = im
            label = np.zeros((im_pad.shape[0], im_pad.shape[1]), dtype='float32')
            
            if self.normalization == 'regular':
                im_pad = im_pad * 1.0 / 255
            elif self.normalization == 'centeral':
                im_pad = (im_pad - 128) * 1.0 / 128
            else:
                pass
            num_segments = np.ceil(im_pad.shape[0] * im_pad.shape[1] / self.u_value)
            segments = slic(img_as_float(im_pad), n_segments=num_segments, sigma=self.std_value,
                            compactness=self.compactness_value)
            # get center location
            props_dict = regionprops_table(segments, properties=['label', 'area', 'centroid'])
            props_df = pd.DataFrame(props_dict)

            props_df['bb_start_x'] = props_df['centroid-1'] - int(patch_size / 2)
            props_df['bb_start_y'] = props_df['centroid-0'] - int(patch_size / 2)
            r, c = im_pad.shape[:2]
            for i, row in props_df.iterrows():
                row_start, row_end = row['bb_start_y'], row['bb_start_y'] + patch_size,
                col_start, col_end = row['bb_start_x'], row['bb_start_x'] + patch_size,
                if any([row_start < 0, row_end > r, col_end > c, col_start < 0]):
                    # print('boundary superpixel; skipped')
                    continue
                patch = im_pad[row_start: row_end, col_start: col_end, :]
                pred_probabilities = model.predict(np.expand_dims(patch, axis=0))[0]
                rows, cols = np.where(segments == row['label'])
                label[rows, cols] = np.argmax(pred_probabilities)
            
            label = label[pad_size:im.shape[0] + pad_size, pad_size:im.shape[1] + pad_size]
            
            io.imsave(seg_img_full_path, label.astype('uint8'))
    
    def apply_multiprocessing(self):
        
        for batch_name in os.listdir(self.input_dir):
            
            path_ = os.path.join(self.input_dir, batch_name)
            # there are some files created by windows system that starts with .
            if batch_name.startswith('.') is True or not os.path.isdir(path_):
                print('{} found and continue to next slide/batch')
                continue
            # check if image name has all image name patterns
            file_names_list = [image_name for image_name in os.listdir(path_) if
                               any([p in image_name for p in self.img_name_pattern]) and
                               image_name.startswith('.') is False]

            if self.multi_process is True:
                print('multiprocessing is used')
                if len(file_names_list) < self.num_processes:
                    num_processes = len(file_names_list)
                else:
                    num_processes = self.num_processes
                    
                num_elem_per_process = int(np.ceil(len(file_names_list) / num_processes))
                
                file_names_list_list = []
                
                for i in range(num_processes):
                    start_ = i * num_elem_per_process
                    x = file_names_list[start_: start_ + num_elem_per_process]
                    file_names_list_list.append(x)
                
                print('{} processes created.'.format(num_processes))
                # create list of processes
                processes = [
                    mp.Process(target=self.eval_tiles, args=(batch_name,
                                                             file_names_list_list[process_num], process_num)) for
                    process_num in range(num_processes)]
                
                # Run processes
                for p in processes:
                    p.start()
                
                # Exit the completed processes
                for p in processes:
                    p.join()
                print('All Processes finished!!!')
                
            else:
                print('running without multiprocessing. If you want to use multiprocessing, set multi_process=True')
                self.eval_tiles(batch_name, file_names_list, 0)

    def run_multi_process(self, batch_name, img_files_names_list):
    
        n = len(img_files_names_list)
    
        if n < self.num_processes:
            self.num_processes = n
    
        num_elem_per_process = int(np.ceil(n / self.num_processes))
    
        file_names_list_list = []
    
        for i in range(self.num_processes):
            start_ = i * num_elem_per_process
            file_names_list_list.append(img_files_names_list[start_: start_ + num_elem_per_process])
    
        # create list of processes
        processes = [
            mp.Process(target=self.eval_tiles, args=(batch_name, file_names_list_list[process_num], process_num))
            for
            process_num in range(self.num_processes)]
    
        print('{} processes created'.format(self.num_processes))
    
        # Run processes
        for p in processes:
            p.start()
    
        # Exit the completed processes
        for p in processes:
            p.join()
        print('All Processes finished!!!')

    def run(self):
        # if self.run_on_batch:
        # batch_names = [batch_name for batch_name in os.listdir(self.input_dir)]
        # print('N={}, Batch names:{}'.format(len(batch_names), batch_names))
        # batch_names.sort()
    
        # if self.batch_index > len(batch_names):
        #     return 0
        #
        # if self.batch_index == -1:
        #     batches = batch_names
        # else:
        #     batches = [batch_names[self.batch_index - 1]]
        #
        # print('processing batch:{}'.format(batches))
        # for batch_name in batches:
        print(self.img_name_pattern)
        if not os.path.isdir(os.path.join(self.input_dir, self.slide_name)):
            return 0
        # in cws directory there are some files starting with "."
        img_files_list_all = [f for f in os.listdir(os.path.join(self.input_dir,
                                                                 self.slide_name)) if not f.startswith('.')]
        if self.img_name_pattern is None:
            print('not checking pattern')
            img_files_names_list = img_files_list_all
        else:
            print('checking pattern')
            img_files_names_list = [file_name for file_name in
                                    img_files_list_all if any([x in file_name for x in self.img_name_pattern])
                                    is True]
        print(img_files_names_list)
        if self.num_processes > 1:
            self.run_multi_process(batch_name=self.slide_name, img_files_names_list=img_files_names_list)
        else:
            self.eval_tiles(self.slide_name, img_files_names_list, 0)
    
        # else:
        #     batch_name = os.path.basename(self.input_dir)
        #     print('Processing images in folder : {}'.format(batch_name))
        #     # in cws directory there are some files starting with "."
        #     img_files_list_all = [f for f in os.listdir(os.path.join(self.input_dir,
        #                                                              batch_name)) if not f.startswith('.')]
        #
        #     if self.img_name_pattern is None:
        #         img_files_names_list = img_files_list_all
        #     else:
        #         img_files_names_list = [file_name for file_name in
        #                                 img_files_list_all if any([x in file_name for x in self.img_name_pattern])
        #                                 is True]
        #
        #     if self.multi_process:
        #         self.run_multi_process(batch_name=batch_name, img_files_names_list=img_files_names_list)
        #     else:
        #         self.eval_tiles(batch_name, img_files_names_list, 0)

if __name__ == '__main__':
    
    output_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\2021-01-25\data\Result'
    input_dir = r'Y:\yhagos\Myeloma_BM_mapping\raw-data\20201026_diagnostic_panel1\cws'
    model_dir = r'Y:\yhagos\Myeloma_BM_mapping\Data\TS\2021-01-25\data\Polyscope_rect\train_val\BestModel'
    
    # for folder in os.listdir(checkpoints_filepath):
    
    params = dict(model_weight_dir=os.path.join(model_dir,  'best_model.h5'),
                  input_dir=input_dir,
                  output_dir=os.path.join(output_dir, 'MoSaicNet_Pipeline'),
                  normalization='regular',
                  num_processes=8,
                  img_name_pattern=['Da'],
                  multi_process=True
                  )
    
    obj = TissueSegmentation(**params)
    obj.apply_multiprocessing()
    
    print('done')
