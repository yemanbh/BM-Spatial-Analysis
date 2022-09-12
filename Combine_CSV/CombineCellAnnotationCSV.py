import pandas as pd
import numpy as np
import os
import pickle
import pathlib
import json


class CombineCellAnnotationCSV(object):
    def __init__(self,
                 csv_files_dir,
                 cws_dir,
                 output_dir,
                 slide_name,
                 down_sample_factor=1
                 ):
        self.csv_files_dir = str(pathlib.Path(csv_files_dir))
        self.cws_dir = str(pathlib.Path(cws_dir))
        self.output_dir = str(pathlib.Path(output_dir))
        self.down_sample_factor = down_sample_factor
        self.slide_name = slide_name
        os.makedirs(output_dir, exist_ok=True)

    def combineCSVfiles(self):
        # for slide in os.listdir(self.csv_files_dir):
        # print(self.slide_name)
        # slide_name = os.path.splitext(slide)[0]
        param = pickle.load(open(os.path.join(self.cws_dir, self.slide_name, 'param.p'), 'rb'))
        slide_dimension = np.array(param['slide_dimension']) / param['rescale']
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        cws_read_size = param['cws_read_size']
        cws_h = cws_read_size[0]
        cws_w = cws_read_size[1]

        cell_detail = pd.DataFrame(columns=['Class', 'X', 'Y'])
        tile_iteration = 0
        for h in range(int(np.ceil(slide_h / cws_h))):
            for w in range(int(np.ceil(slide_w/cws_w))):
                start_h = h * cws_h
                start_w = w * cws_w
                file_name = os.path.join(self.csv_files_dir, self.slide_name, 'Da'+str(tile_iteration)+'.csv')
                print(file_name)
                assert os.path.exists(file_name)
                # increment here becuase there might be empty csv
                tile_iteration += 1
                df = pd.read_csv(file_name)
                if len(df) == 0:
                    # print(file_name, '\n is empty')
                    continue
                df = df[['X', 'Y', 'Class']]
                df.columns = ['X', 'Y', 'Class']
                df['X'] = df['X'] + start_w
                df['Y'] = df['Y'] + start_h
                cell_detail = cell_detail.append(df, sort=False)
                
                # except Exception:
                #     print(file_name, '\n is empty')
                #     continue
                # print(file_name)
        print('slide:{} , last tile:{}, number of tile:{}'.format(self.slide_name, 'Da'+str(tile_iteration-1), tile_iteration))

        # down sample it
        if self.down_sample_factor != 1:
            cell_detail['X'] = np.round(np.float64(cell_detail.loc[:, 'X']) / self.down_sample_factor).astype('int')
            cell_detail['Y'] = np.round(np.float64(cell_detail.loc[:, 'Y']) / self.down_sample_factor).astype('int')
        # save
        cell_detail.to_csv(os.path.join(self.output_dir, self.slide_name +'.csv'), index=False)
    print('DONE!')


def combine_cws_csv(csv_files_dir, cws_dir, slide_name, output_dir, down_sample_factor=1):
    os.makedirs(output_dir, exist_ok=True)
    params = dict(
        csv_files_dir=csv_files_dir,
        cws_dir=cws_dir,
        slide_name=slide_name,
        output_dir=output_dir,
        down_sample_factor=down_sample_factor
    )
    with open(os.path.join(output_dir, 'combine_info.json'), 'w') as j_file:
        json.dump(params, fp=j_file, indent=3)
    obj = CombineCellAnnotationCSV(**params)
    obj.combineCSVfiles()
    
if __name__ == '__main__':
    output_dir = r'Z:\yhagos\Projects\Myeloma_BM_mapping\Results\20201118_CD_CC_Panel1\results\Combined_csv_fp_removed'
    params = {
        'csv_files_dir': r'Z:\yhagos\Projects\Myeloma_BM_mapping\Results\20201118_CD_CC_Panel1\results\AnnotatedCellsCoord_fp_removed',
        'cws_dir': r'Z:\yhagos\Projects\Myeloma_BM_mapping\data\20201026_diagnostic_panel1\cws',
        'output_dir': output_dir,
        'down_sample_factor': 1
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'combine_info.json'), 'w') as j_file:
        json.dump(params, fp=j_file, indent=3)
    obj = CombineCellAnnotationCSV(**params)
    obj.combineCSVfiles()







