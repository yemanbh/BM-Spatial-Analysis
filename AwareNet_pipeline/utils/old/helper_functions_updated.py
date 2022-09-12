import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

def remove_overlapping_detection(df, distance_threshold):

    """
    Note: this code removes overlapping prediction and return the corrected dataframe
    """
    print('     * remove over detection.')

    assert all([a in df.columns for a in ['X', 'Y']]), 'input dataframe should have X, and Y coordinates.' \
                                                       'df[columns]={}'.format(df.columns)
    if 'Probability' not in df.columns:
        df['Probability'] = [1.0]*len(df)

    X = np.array(df['X'])
    Y = np.array(df['Y'])

    xy = np.stack((X, Y), axis=-1)

    dist = euclidean_distances(xy, xy)
    # to remove diagonal fill them with a large value, 1000 chosen here
    np.fill_diagonal(dist, 1000.0)
    # print(dist)
    dst_thresh = (dist < distance_threshold) * 1
    # print(dist)
    row, _ = np.where(dist < distance_threshold)
    nn_row = list(np.unique(row))

    tp = ['Y'] * dist.shape[0]  # mark every cell as true positive prediction

    # for every cell with a nearby cell (less than distance threshold), compare values based on cell prediction area and
    #  cell prediction probability, select the most likely cell from the cell under consideration and neighbouring cells
    area_pro = np.multiply(df['Probability'].to_numpy(), df['Area'].to_numpy())
    area_pro = np.repeat(np.reshape(area_pro, (1, len(area_pro))), repeats=len(df), axis=0)

    # mask out non neighbouring cells,
    # update distance threshold matrix to include diagonal
    np.fill_diagonal(dst_thresh, 1.0)
    area_pro_dist = np.multiply(area_pro, dst_thresh)

    for r in nn_row:

        if tp[r] == 'Y':
            arg_max = np.argmax(area_pro_dist[r, :])  # index of the true cell
            cols = np.where(area_pro_dist[r, :])[0]

            # del nn_cols[arg_max]
            # tp[nn_cols] = 'N'
            for col_index in list(cols):
                if col_index != arg_max:
                    tp[col_index] = 'N'
    df['tp'] = tp

    df = df.loc[df['tp'] == 'Y', :]

    df = df.drop(columns=['tp'])

    # return corrected predictions
    return df

def mark_cell_center(im, df, file_name, cell_2_color_dict=None):
    dpi = 100

    height, width, nbands = im.shape

    # What size does the figure need to be in inches to fit the image?
    fig_size = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im, interpolation='nearest')
    
    # display cells on the image
    if cell_2_color_dict is None:
        sns.scatterplot(x='X', y='Y', data=df, hue='Class', ax=ax, edgecolor=None, s=0.5)
    else:
        sns.scatterplot(x='X', y='Y', data=df, hue='Class', palette=cell_2_color_dict, ax=ax, edgecolor=None)

    # Ensure we're displaying with square pixels and the right extent.
    # This is optional if you haven't called `plot` or anything else that might
    # change the limits/aspect.  We don't need this step in this case.
    ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)

    # save image with cell annotation
    fig.savefig(file_name, dpi=dpi, transparent=True)