import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def remove_overlapping_detection(df,
                                 distance_threshold):

    """
    Note: this code removes overlapping prediction and return the corrected dataframe
    """
    print('     * remove over detection.')

    assert all([a in df.columns for a in ['X', 'Y']]), 'input dataframe should have X, and Y coordinates.' \
                                                       'df[columns]={}'.format(df.columns)
    if 'p' not in df.columns:
        df['p'] = [1.0]*len(df)

    #
    X = np.array(df['X'])
    Y = np.array(df['Y'])

    xy = np.stack((X, Y), axis=-1)

    dist = euclidean_distances(xy, xy)
    np.fill_diagonal(dist, 1000.0)
    dst_thresh = (dist < distance_threshold) * 1

    row, col = np.where(dist < distance_threshold)

    row = list(row)

    col = list(col)

    # row_ = [v for i, v in enumerate(row) if row[i] != col[i]]

    tp = ['Y'] * dist.shape[0]  # true positive prediction

    p = df['p'].values

    area = df['Area'].values

    for i in row:

        if tp[i] == 'Y':

            cols = np.where(dst_thresh[i, :])[0]
            probabilities = p[cols]
            pred_map_areas = area[cols]

            argmax = np.argmax(probabilities * pred_map_areas)

            for c, value in enumerate(list(cols)):

                if c != argmax:
                    tp[value] = 'N'
    df['tp'] = tp

    df = df.loc[df['tp'] == 'Y', :]

    df = df.drop(columns=['tp'])

    # return corrected predictions
    return df