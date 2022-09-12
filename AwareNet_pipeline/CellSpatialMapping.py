# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:45:31 2018
@author: yhagos
"""
from skimage import io
import cv2
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
import time
from tensorflow import keras

from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.transform import rescale
from skimage.morphology import dilation, disk


from AwareNet_pipeline.utils.helper_functions_new_updated import remove_overlapping_detection, mark_cell_center
from AwareNet_pipeline.CellClassification import CellClassification
from skimage import measure


class DetectCells(object):

    def __init__(self,
                 cell_detection_model_dir,
                 input_dir,
                 output_dir,
                 cell_names_ordered,
                 scale,
                 slide_name,
                 num_processes=1,
                 run_on_batch=True,
                 pred_probability_threshold=0.8,
                 split_area_threshold=0.95,
                 distance_threshold=10.0,
                 prediction_prob_area_threshold=20,
                 cell_classification_model_dir=None,
                 img_name_pattern=None,  # a list of file name patterns
                 cell_label_text=None,
                 cws_mask=None,
                 count_loss_used=False,
                 normalize='regular',
                 save_annotated_image=False,
                 overlap=False,
                 postprocess=False,
                 do_classification=True,
                 save_detection_prob_map=False,
                 ):
        """
        split_area_threshold (T):
            *  if T: it will be taken as threshold value, for example, split_area_threshold = 80
            * if T < 0: it will be treated percentile, for example, split_area_threshold = 0.8
              this means set threshold at 80th percentile
        Cell location will in the scale provided, so further analysis on the original resolution is needed, you should
        rescale them back to original externally.

        mask is not implemented

        """
        self.cell_detection_model_dir = cell_detection_model_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cws_mask = cws_mask
        self.img_name_pattern = img_name_pattern
        self.num_processes = num_processes
        self.scale = scale
        self.pred_probability_threshold = pred_probability_threshold

        self.split_area_threshold = split_area_threshold
        self.distance_threshold = distance_threshold
        self.prediction_prob_area_threshold = prediction_prob_area_threshold
        self.overlap = overlap
        self.normalize = normalize

        self.count_loss_used = count_loss_used

        self.postprocess = postprocess
        self.save_annotated_image = save_annotated_image

        self.cell_names_ordered = cell_names_ordered
        self.cell_label_text = cell_label_text
        self.save_detection_prob_map = save_detection_prob_map
        self.run_on_batch = run_on_batch
        self.slide_name = slide_name
        # self.batch_index = batch_index

        'cell classification initialization'

        self.cell_classification_model_dir = cell_classification_model_dir
        self.do_classification = do_classification

        os.makedirs(self.output_dir, exist_ok=True)

        if self.do_classification is False:
            print('*'*40)
            print('only cell detection will be applied')
            print('*' * 40)
        else:
            assert self.cell_classification_model_dir is not None, 'specify path to classifier model'

    def evaluate_tiles(self, batch_name, img_files_names_list, p_n):

        # get model
        cell_detection_model = self.get_cell_detection_model()

        # set parameters
        input_shape = cell_detection_model.layers[0].input_shape[0]
        self.set_patch_size(input_shape)
        self.set_stride(input_shape)

        # create output folders
        annotated_cells_coord_folder = os.path.join(self.output_dir, 'AnnotatedCellsCoord', batch_name)
        os.makedirs(annotated_cells_coord_folder, exist_ok=True)

        if self.save_annotated_image:
            annotated_tiles_output_folder = os.path.join(self.output_dir, 'AnnotatedTiles', batch_name)
            os.makedirs(annotated_tiles_output_folder, exist_ok=True)
        else:
            annotated_tiles_output_folder = None
            
        if self.save_detection_prob_map:
            cell_detection_mask_folder = os.path.join(self.output_dir, 'CellDetectionMask', batch_name)
            os.makedirs(cell_detection_mask_folder, exist_ok=True)
        else:
            cell_detection_mask_folder = None

        input_im_folder = os.path.join(self.input_dir, batch_name)
        time_elapsed_df = pd.DataFrame(columns=['sample_name', 'tile_name', 'time_elapsed(min)'])
        
        for jj, img_file_name in enumerate(img_files_names_list):
            print(img_file_name)
            time_elapsed_row = [batch_name, img_file_name]
            start_time = time.time()

            print(' + file name:{}, file index:{}/{}'.format(img_file_name,
                                                             jj + 1,
                                                             len(img_files_names_list)))

            # output file names
            files_to_check = []
            elapsed_time_csv_name = os.path.join(self.output_dir, batch_name + '_time_elapsed.csv')

            name_ = os.path.splitext(img_file_name)[0]
            if self.save_annotated_image:
                annotated_im_name = os.path.join(annotated_tiles_output_folder, name_ + '.jpg')
                files_to_check.append(annotated_im_name)
            else:
                annotated_im_name = None
                
            if self.save_detection_prob_map:
                cell_detection_map_name = os.path.join(cell_detection_mask_folder, name_ + '.jpg')
                files_to_check.append(cell_detection_map_name)
            else:
                cell_detection_map_name = None
                
            csv_filename = os.path.join(annotated_cells_coord_folder, name_ + '.csv')
            files_to_check.append(csv_filename)
            
            all_files_exist = all([os.path.isfile(file_path) for file_path in files_to_check])



            if all_files_exist:
                print('     * file already exists in the output folder')
                continue
            else:
                print('     * detecting cells')

            # empty csv file to store location of detected cells
            df = pd.DataFrame(columns=['X', 'Y', 'Area'])

            im = io.imread(os.path.join(input_im_folder, img_file_name))
            if self.scale == 1:
                pass
            else:
                im = rescale(image=im, scale=self.scale, order=0, multichannel=True)

                # image as uint8
                im = (255 * im).astype('uint8')

            'cell mask'
            if self.cws_mask is not None:

                im_mask = io.imread(os.path.join(self.cws_mask, batch_name, img_file_name))
                
                if im_mask.ndim == 3:
                    im_mask = im_mask[:, :, 0]

                if np.sum(im_mask) == 0:
                    if self.save_annotated_image:
                        print('Background tile found')
                        # print(annotated_im_name, im.shape)
                        io.imsave(annotated_im_name, im)
                    df['X'] = []
                    df['Y'] = []
                    df['Area'] = []
                    df['Class'] = []
                    df.to_csv(csv_filename, index=False)

                    continue
                else:
                    pass
            else:
                im_mask = None

            pad_size = 2 * self.patch_size
            n = 2 * pad_size
            im_pad = np.zeros((n + im.shape[0], n + im.shape[1], 3), dtype='uint8')
            
            if self.cws_mask is not None:
                mask_pad = np.zeros((n + im_mask.shape[0], n + im_mask.shape[1]), dtype='uint8')
            else:
                mask_pad = None
                
            n = int(n / 2)
            im_pad[n:im.shape[0] + n, n:im.shape[1] + n, :] = im
            
            if self.cws_mask is not None:
                mask_pad[n:im_mask.shape[0] + n, n:im_mask.shape[1] + n] = im_mask
                
            label = np.zeros(im_pad.shape[:2])

            if self.normalize == 'regular':
                
                im_pad = im_pad * 1.0 / 255
                
            elif self.normalize == 'central':
                
                im_pad = (im_pad - 128) * 1.0 / 128
            else:
                raise Exception('Allowed method names are regular and central')

            padding_err = 12
            shift = int(pad_size / 2) - padding_err

            row_end = im_pad.shape[0] - int(self.patch_size / 2)+padding_err
            col_start_ = shift

            col_end = im_pad.shape[1] - int(self.patch_size / 2)+padding_err
            r = shift

            while r < row_end:
                c = col_start_

                while c < col_end:

                    r_start = r - shift
                    c_start = c - shift
                    
                    p_image = im_pad[r_start:r_start + self.patch_size,
                              c_start:c_start + self.patch_size, :
                              ]

                    if self.cws_mask is not None:
                        p_mask = mask_pad[r_start:r_start + self.patch_size,
                                 c_start:c_start + self.patch_size]
                        
                        if np.sum(p_mask) == 0:
                            c = c + self.stride - 2 * padding_err
                            continue
                            
                    p_image = np.expand_dims(p_image, axis=0)

                    if self.count_loss_used is True:
                        pred, cell_count = cell_detection_model.predict(p_image)
                    else:
                        pred = cell_detection_model.predict(p_image)

                    pred = np.squeeze(pred)
                    # print(pred.shape)
                    pred_val = pred[padding_err:pred.shape[0]-padding_err,
                               padding_err:pred.shape[1]-padding_err]
                    label[r_start + padding_err:r_start + self.patch_size-padding_err,
                        c_start + padding_err:c_start + self.patch_size - padding_err] = pred_val

                    c = c + self.stride - 2 * padding_err

                r = r + self.stride-2*padding_err

            del im_pad

            label = label[n:im.shape[0] + n, n:im.shape[1] + n]

            # get x and y location, and area of cell probability map
            df = self.get_cell_center(label)
            # Save cell detection probability map

            if self.save_detection_prob_map is True:

                detection_map_img = (255 * label).astype('uint8')
                # save original probability as numpy
                # dir_name, f_name = os.path.dirname(cell_detection_map_name), os.path.basename(cell_detection_map_name)
                # np.save(os.path.join(dir_name, os.path.splitext(f_name)[0] + '.npy'), label)
    
                io.imsave(cell_detection_map_name, detection_map_img)

            if self.do_classification is True and len(df) != 0:
                classifier = CellClassification(model_path=self.cell_classification_model_dir,
                                                cell_names_ordered=self.cell_names_ordered,
                                                cell_label_text=self.cell_label_text,
                                                normalize=True,
                                                save_annotated_image=self.save_annotated_image,
                                                file_name=annotated_im_name
                                                )
                df = classifier.classify_cells(image_in=im, input_df=df)
            else:
                # remove overlapping prediction
                if len(df) != 0:
                    df = remove_overlapping_detection(df=df,  distance_threshold=self.distance_threshold)

                # save annotated image
                if self.save_annotated_image:
                    mark_cell_center(im, df, file_name=annotated_im_name)

            end_time = time.time()
            t = end_time - start_time
            time_elapsed = np.round(t, decimals=2)

            time_elapsed_row.append('{0:.2f}'.format(time_elapsed / 60))
            time_elapsed_df.loc[len(time_elapsed_df)] = time_elapsed_row

            # cell detection csv file
            df.to_csv(csv_filename, index=False)

            if os.path.isfile(elapsed_time_csv_name):
                df_old = pd.read_csv(elapsed_time_csv_name)
                time_elapsed_df = pd.concat([df_old,
                                             time_elapsed_df],
                                            axis=0,
                                            ignore_index=True,
                                            sort=False)
                time_elapsed_df.to_csv(elapsed_time_csv_name, index=False)

    def set_patch_size(self, model_input_shape):
            self.patch_size = model_input_shape[1]

    def set_stride(self, model_input_shape):
        self.stride = model_input_shape[1]

    def get_cell_detection_model(self):

        model = keras.models.load_model(self.cell_detection_model_dir, compile=False)

        return model

    def get_cell_center(self, im):
    
        print('     * getting cell center from probability map ')
        df = pd.DataFrame(columns=['X', 'Y', 'Area', 'Label'])
        # binarize image and preprocess
        im = (im > self.pred_probability_threshold) * 1
        im = binary_fill_holes(im)
    
        # binary image labeling
        labeled_im = measure.label(input=im, connectivity=im.ndim)

        del im
        # in skimage '0.16.2' if there are no objects in labeled image, it throw an error
        u_labels_ = np.unique(labeled_im)
        if len(u_labels_) == 1:
            print('Image without cells found and returning empty csv file')
            return df
        else:  # run the following pipeline
            pass
        props = measure.regionprops_table(label_image=labeled_im, properties=('label', 'area', 'centroid'))
    
        df['X'] = props['centroid-1'].tolist()
        df['Y'] = props['centroid-0'].tolist()
    
        df['Area'] = props['area'].tolist()
        df['Label'] = props['label'].tolist()
    
        if self.postprocess is True:
        
            if self.split_area_threshold < 1:
                split_area_threshold = np.percentile(df['Area'].to_numpy(), 100 * self.split_area_threshold)
            else:
                split_area_threshold = self.split_area_threshold
        
            print('         * using {} percentile as split_area_threshold = {}'.format(100 * self.split_area_threshold,
                                                                                       split_area_threshold))
            # large objects
        
            large_obj_df = df.loc[df['Area'] >= split_area_threshold].reset_index(drop=True)
            # small objects
            # remove very small areas
            small_obj_df = df.loc[
                ((df['Area'] < split_area_threshold) & (df['Area'] > self.prediction_prob_area_threshold))].reset_index(
                drop=True)
        
            large_obj_image = (np.isin(labeled_im, large_obj_df['Label'].to_list())) * 1
        
            # split large probability maps
            separated_objects_df = self.split_cells(large_obj_image)
        
            df_merged = pd.concat([small_obj_df, separated_objects_df], axis=0, ignore_index=True, sort=False)
        
            return df_merged
    
        else:
        
            return df

    @staticmethod
    def split_cells(x):
    
        df = pd.DataFrame(columns=['X', 'Y', 'Area', 'Label'])
    
        distance = ndi.distance_transform_edt(x)
    
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7, 7)), labels=x)
        local_maxi = local_maxi * 1
        local_maxi_dilated = dilation(local_maxi, disk(2))
    
        labeled_im = measure.label(input=local_maxi_dilated, connectivity=local_maxi_dilated.ndim)
        
        # in skimage '0.16.2' if there are no objects in labeled image, it throw an error
        u_labels_ = np.unique(labeled_im)
        if len(u_labels_) == 1:
            print('Image without cells found and returning empty csv file')
            return df
        else:  # run the following pipeline
            pass
        
        del local_maxi_dilated
        props = measure.regionprops_table(label_image=labeled_im, properties=('label', 'centroid', 'area'))
    
        df['X'] = props['centroid-1'].tolist()
        df['Y'] = props['centroid-0'].tolist()
        df['Area'] = props['area'].tolist()
        df['Label'] = props['label'].tolist()
    
        return df

    @staticmethod
    def put_markers_on_binary_image(X, Y, im):
        """
        this is removed because it is time consuming
        :param X:
        :param Y:
        :param im:
        :return:
        """
        r = 3
        for i in range(len(X)):
            cv2.circle(im, (X[i], Y[i]), r, color=(1, 1, 1), thickness=-1)
    
        return (im > 0.8) * 1

    @staticmethod
    def mark_cell_center_old(im, X, Y):
        # this is slow and it is not used anymore
        print('     * marking cells center')
        r = 2
        X = list(X)  # row
        Y = list(Y)  # col
        for i in range(len(X)):
            cv2.circle(im, (X[i], Y[i]), r, color=(255, 0, 0), thickness=-1)
    
        return im
    
    def run_multi_process(self, batch_name, img_files_names_list ):
        
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
            mp.Process(target=self.evaluate_tiles, args=(batch_name, file_names_list_list[process_num], process_num))
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
        #     batch_names = [batch_name for batch_name in os.listdir(self.input_dir)]
        #     print('N={}, Batch names:{}'.format(len(batch_names), batch_names))
        #     batch_names.sort()
    
        #     if self.batch_index > len(batch_names):
        #         return 0
        #
        #     if self.batch_index == -1:
        #         batches = batch_names
        #     else:
        #         batches = [batch_names[self.batch_index - 1]]
        #
        #     print('processing batch:{}'.format(batches))
        #     for batch_name in batches:
        #
        #         if not os.path.isdir(os.path.join(self.input_dir, batch_name)):
        #             continue
        #         # in cws directory there are some files starting with "."
        #         img_files_list_all = [f for f in os.listdir(os.path.join(self.input_dir,
        #                                                                  batch_name)) if not f.startswith('.')]
        #         if self.img_name_pattern is None:
        #             img_files_names_list = img_files_list_all
        #         else:
        #             img_files_names_list = [file_name for file_name in
        #                                     img_files_list_all if any([x in file_name for x in self.img_name_pattern])
        #                                     is True]
        #         if self.multi_process:
        #             self.run_multi_process(batch_name=batch_name, img_files_names_list=img_files_names_list)
        #         else:
        #             self.evaluate_tiles(batch_name, img_files_names_list, 0)
        #
        # else:
        # batch_name = os.path.basename(self.input_dir)
        # print('Processing images in folder : {}'.format(batch_name))
        # in cws directory there are some files starting with "."
        img_files_list_all = [f for f in os.listdir(os.path.join(self.input_dir,
                                                                  self.slide_name)) if not f.startswith('.')]

        if self.img_name_pattern is None:
            img_files_names_list = img_files_list_all
        else:
            img_files_names_list = [file_name for file_name in
                                    img_files_list_all if any([x in file_name for x in self.img_name_pattern])
                                    is True]

        if self.num_processes > 1:
            self.run_multi_process(batch_name=self.slide_name, img_files_names_list=img_files_names_list)
        else:
            self.evaluate_tiles(self.slide_name, img_files_names_list, 0)


if __name__ == '__main__':
    pass

