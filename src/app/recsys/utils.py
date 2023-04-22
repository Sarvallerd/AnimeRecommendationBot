import numpy as np
import pandas as pd


def get_dists(DATA_PATH: str, similarity):
    vectors = np.load(DATA_PATH + 'matrix.npz')['arr_0']
    return similarity(vectors)


def get_recommendations(matrix, df, name):
    index = df[df['Name'] == name].index[0]
    cluster = df[df['Name'] == name]['cluster'].values[0]
    result = list(enumerate(matrix[index]))
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)[1:100]
    rec_idx = [item[0] for item in sorted_result]
    rec_df = df.loc[rec_idx].copy().reset_index(drop=True)
    rec_df = rec_df.loc[rec_df['cluster'] == cluster]
    rec_df = rec_df.sort_values(by=['Score', 'year'], ascending=False).reset_index(drop=True)
    return rec_df
