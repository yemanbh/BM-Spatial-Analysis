import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats


class PointPatternAnalysis(object):
    """
    Point pattern analysis

        * it generates the number of event_cell at a distance (d) from reference cells for an array of d values.

        * spatial_data: this should be a dataframe with X, Y, and Class columns.
        * reference_cell and event_cell should be list of element of class names in class column.
        * if mpp is provided pixel locations are converted to micros to compute distance
    """
    def __init__(self,
                 spatial_data_dir: str,
                 reference_cell: tuple,
                 event_cell: tuple,
                 output_dir: str,
                 distances_list: tuple,
                 cell_names=None,
                 mpp=None,

                 ):

        self.spatial_data_dir = spatial_data_dir
        self.distances_list = distances_list
        self.reference_cell = reference_cell
        self.event_cell = event_cell
        self.mpp = mpp
        self.cell_names = cell_names

        folder = "#{}_Within_D_from_{}".format('_and_'.join(event_cell), '_and_'.join(reference_cell) )
        self.output_dir_count = os.path.join(output_dir, 'proximity_analysis', folder)
        # self.output_dir_percentage = os.path.join(output_dir, 'nn_percentage', folder)
        # os.makedirs(self.output_dir_percentage, exist_ok=True)
        os.makedirs(self.output_dir_count, exist_ok=True)

    def count_cells_within_distance(self) -> pd.DataFrame():
        print('computing #{} cells within distance (d) from {} cells'.format(self.event_cell, self.reference_cell))
        if self.cell_names is None:
            event_cells = self.event_cell
        else:
            event_cells = list(set(self.event_cell + self.cell_names))
        count_df = pd.DataFrame()
        # nn_dist_df = pd.DataFrame(columns=['SlideName', 'RefCellType'] + event_cells)
        for file_name in os.listdir(self.spatial_data_dir):
            # print(file_name)
            slide_name = os.path.splitext(file_name)[0]
            if not any([file_name.endswith(p) for p in ['csv', 'xlsx']]):
                # print('not spatial data file found; skipped')
                continue

            if file_name.endswith('xlsx'):
                df_in = pd.read_excel(os.path.join(self.spatial_data_dir, file_name))
            else:
                df_in = pd.read_csv(os.path.join(self.spatial_data_dir, file_name))

            # unit conversion
            if self.mpp is not None:
                # print('converting  spatial data to microns')
                df_in[['X', 'Y']] = df_in[['X', 'Y']].astype('float32') * self.mpp
            else:
                pass
                # print('raw spatial data will be used.')

            # spatial_data_row = [os.path.splitext(file_name)[0]]
            # percentage_row = [os.path.splitext(file_name)[0]]
            df_count_temp = pd.DataFrame()
            # nn_distance_list = []
            for event_cell in event_cells:
                reference_cell_xy = df_in.loc[df_in['Class'].isin(self.reference_cell), ['X', 'Y']].to_numpy()
                event_cell_xy = df_in.loc[df_in['Class'] == event_cell, ['X', 'Y']].to_numpy()
                all_xy = df_in[['X', 'Y']].to_numpy()
                # compute Euclidean distance from event 1 to event 2
                if len(reference_cell_xy) == 0 or len(event_cell_xy) == 0:
                    print(f'either {self.reference_cell} or {self.event_cell} is missing in the WSI, {file_name}')
                    print('image skipped')
                    continue
                    
                # compute Euclidean distance from event 1 to event 2
                cell_distance = cdist(reference_cell_xy, event_cell_xy, metric='euclidean')
                dist_all = cdist(reference_cell_xy, all_xy)
                if self.reference_cell[0] == self.event_cell:
                    np.fill_diagonal(cell_distance, np.inf)
                    # np.fill_diagonal(cell_distance, np.inf)
                cell_count_list = []
                for dist in self.distances_list:
                    # number of cells within dist from reference cell
                    mean_cell_count = np.mean(np.sum(cell_distance <= dist, axis=1))    # float value
                    cell_count_list.append(mean_cell_count)
                    # count percentage of reference cell which have event cell within a distance
                    # percentage = sum(min_d <= dist) / min_d.__len__()
                    # percentage_row.append(percentage)
                # update
                df_count_temp[event_cell] = cell_count_list
            
            # update nebourhood distance data
            df_count_temp['SlideName'] = slide_name
            df_count_temp['RefCellType'] = self.reference_cell[0]
            df_count_temp[self.cell_names] = df_count_temp[self.cell_names].div(df_count_temp[self.cell_names].sum(axis=1), axis=0)
            df_count_temp['Distance'] = self.distances_list
            count_df = pd.concat([count_df, df_count_temp], axis=0, sort=False)
        # save files
        count_df.to_csv(os.path.join(self.output_dir_count, 'mean_nearest_cells_count.csv'), index=False)

        return count_df

if __name__ == '__main__':

        output_dir = r'output'
        combined_csv_dir = r'sample_data'

        cell_pairs = [['CD4', 'FOXP3+CD4+'], ['CD8', 'FOXP3+CD4+']]
        cell_names_panel = ['FOXP3+CD4+', 'CD4', 'CD8']

        for cell_pair in cell_pairs:
            params = dict(spatial_data_dir=combined_csv_dir,
                          reference_cell=[cell_pair[0]],
                          event_cell=[cell_pair[1]],
                          output_dir=output_dir,
                          cell_names=cell_names_panel,
                          mpp=0.4606,
                          distances_list=[30, 50, 100, 150, 200, 250, 300])
            pp = PointPatternAnalysis(**params)
            pp.count_cells_within_distance()
        print('DONE')
