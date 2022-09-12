import pandas as pd
import os
import matplotlib.pyplot as plt


def count_cells_cws_combined(input_dir, output_dir, cell_names, class_col_name):
    os.makedirs(output_dir, exist_ok=True)
    all_slides_cell_count_df = pd.DataFrame(columns=['SlideName'] + cell_names)

    for ii, file_name in enumerate(os.listdir(input_dir)):
        if not file_name.endswith('csv'):
            continue
        print('processing slide: ', file_name)
        row = [file_name.replace('.csv', '')]

        # read file
        df = pd.read_csv(os.path.join(input_dir, file_name))
        if len(df) is 0:
            continue
        # count cells
        print(df.head())
        slide_all_slides_cell_count_df = df[class_col_name].value_counts().to_frame().transpose()

        # for missing cells update count with 0
        counts = []
        for cell in cell_names:
            if cell in slide_all_slides_cell_count_df.columns:
                counts.append(slide_all_slides_cell_count_df[cell].values[0])
            else:
                counts.append(0)

        # update count data frame
        row = row + counts
        all_slides_cell_count_df.loc[len(all_slides_cell_count_df)] = row
    
    # save file
    all_slides_cell_count_df.to_csv(os.path.join(output_dir, 'cells_count.csv'), index=False)
    
    # plot data distribution
    # original cell count
    all_slides_cell_count_df.plot(kind='bar', stacked=True, x='SlideName')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_count_unnormalized.png'), dpi=600)
    # normalized cell count
    slide_name_df = all_slides_cell_count_df[['SlideName']]
    counts_df = all_slides_cell_count_df.drop(columns=['SlideName'])
    normalized_counts_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    normalized_counts_df = pd.merge(slide_name_df, normalized_counts_df, left_index=True, right_index=True)
    normalized_counts_df.plot(kind='bar', stacked=True, x='SlideName')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_count_normalized.png'), dpi=600)
    plt.show()


def count_cells_cws_not_combined(input_dir, output_dir, cell_names, class_col_name, excluee_from_figure=None):
    
    os.makedirs(output_dir, exist_ok=True)
    col_names = ['SlideName'] + cell_names
    all_slides_cell_count_df = pd.DataFrame(columns=col_names)

    for ii, slide_name in enumerate(os.listdir(input_dir)):
        print('processing slide: ', slide_name)
        # read cws cell detection
        df_cws_combined = pd.DataFrame()
        for file_name in os.listdir(os.path.join(input_dir, slide_name)):
            df = pd.read_csv(os.path.join(input_dir, slide_name, file_name))
            
            df_cws_combined = pd.concat([df_cws_combined, df], axis=0, ignore_index=True, sort=False)
        if len(df_cws_combined) is 0:
            print(f'{slide_name} does not have cells')
            continue
        # count cells
        slide_all_slides_cell_count_df = df_cws_combined[class_col_name].value_counts().to_frame().transpose()
        
        # for missing cells update count with 0
        counts = []
        for cell in cell_names:
            if cell in slide_all_slides_cell_count_df.columns:
                counts.append(slide_all_slides_cell_count_df[cell].values[0])
            else:
                counts.append(0)
        
        # update count data frame
        new_record = dict(zip(col_names, [slide_name.replace('.csv', '')] + counts))
        # all_slides_cell_count_df.loc[len(all_slides_cell_count_df)] = row
        all_slides_cell_count_df = all_slides_cell_count_df.append(new_record, ignore_index=True, sort=False)
    
    # save file
    all_slides_cell_count_df.to_csv(os.path.join(output_dir, 'cells_count_all.csv'), index=False)
    
    # plot data distribution
    if excluee_from_figure is not None:
        all_slides_cell_count_df = all_slides_cell_count_df.loc[~all_slides_cell_count_df['SlideName'].isin(excluee_from_figure)]
    all_slides_cell_count_df.plot(kind='bar', stacked=True, x='SlideName')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_count_all.png'), dpi=600)
    # normalized cell count

if __name__=='__main__':
    # Panel1: ['CD8', 'CD4', 'FOXP3+CD4+']
    #Panel2:['BLIMP1', 'CD4', 'CD8']
    # cell_names = ['BLIMP1', 'CD4', 'PD1', 'PD1+CD4+']
    # cell_names_panel2 = ['BLIMP1', 'CD4', 'CD8']
    cell_names_panel1 = ['CD8+', 'FOXP3-CD4+', 'FOXP3+CD4+']
    # input_dir = r'X:\yhagos\Myeloma_BM_mapping\Data\Resutls\panel2_2021-04-02\AwareNet_pipeline\AnnotatedCellsCoord_fp_removed'
    # output_dir = r'X:\yhagos\Myeloma_BM_mapping\Data\Resutls\panel2_2021-04-02\AwareNet_pipeline\Cell_count_cws'
    # count_cells_cws_not_combined(input_dir=input_dir,
    #                              output_dir=output_dir,
    #                              class_col_name='Class',
    #                              cell_names=cell_names)
    
    # cell_names = ['BLIMP1', 'CD4', 'CD8']
    input_dir = r'X:\yhagos\Myeloma_BM_mapping\Data\Resutls\20201118_CD_CC_Panel1\results\Combined_csv_fp_removed_cell_renamed'
    output_dir = r'X:\yhagos\Myeloma_BM_mapping\Data\Resutls\20201118_CD_CC_Panel1\results\cell_count'
    count_cells_cws_combined(input_dir=input_dir,
                             output_dir=output_dir,
                             class_col_name='Class',
                             cell_names=cell_names_panel1)
