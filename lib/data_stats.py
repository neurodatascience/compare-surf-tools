# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd

#TODO

def cross_correlations(df1,df2,n_roi,subject_ID_col):
    """ Computes correlation between features produced by two pipelines 
        Note: avg cross-correlations using .corr() diangonals because corrwith() doesn't seem to always work... 
    """
    xcorr = 0
    col_rename_dict_1 = {}
    col_rename_dict_2 = {}
    if ((n_roi != len(df1.columns) - 2) | (n_roi != len(df2.columns) - 2)): 
        print('Number of ROIs mismatch with the dataframe size')
    else:
        for col in df1.columns:
            col_rename_dict_1[col] = str(col) + '_df1' 
            col_rename_dict_2[col] = str(col) + '_df2'

        df1 = df1.rename(columns=col_rename_dict_1)
        df2 = df2.rename(columns=col_rename_dict_2)
        df1 = df1.rename(columns={'{}_df1'.format(subject_ID_col):subject_ID_col})
        df2 = df2.rename(columns={'{}_df2'.format(subject_ID_col):subject_ID_col})

        concat_df = df1.merge(df2, on=subject_ID_col)
        print('Shape of concatinated dataframe of two pipelines {}'.format(concat_df.shape))
        corr_mat = concat_df.corr()
        print('Shape of corr mat {}'.format(corr_mat.shape))
        xcorr = corr_mat.values[:n_roi,n_roi:2*n_roi].diagonal()

    return xcorr