# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd

def load_processed_data(csv_dict, col_list, na_action):
    """ Reads CSV outputs from the processed MR images by pipelines such as FreeSurfer, ANTs, CIVET, etc.
    """ 

    n_csv = len(csv_dict)
    print('Number of datasets: {}'.format(n_csv))
    print('Number of columns: {}'.format(len(col_list)))

    df_concat = pd.DataFrame()
    for c in csv_dict.keys():
        print('\nReading {} csv'.format(c))
        csv_data = pd.read_csv(csv_dict[c])
        if check_processed_data(csv_data,col_list,na_action):
            print('Basic CSV check passed')
            csv_data['pipeline'] = np.tile(c,len(csv_data))
            df_concat = df_concat.append(csv_data,sort=True)
            print('Shape of the concat dataframe {}'.format(df_concat.shape))        

    return df_concat

def check_processed_data(df,col_list,na_action):
    """ Checks if provided dataframe consists of required columns and no missing values
    """
    check_passed = True

    # Check columns
    df_cols = df.columns
    if set(df_cols) != set(col_list):
        check_passed = False
        print('Column names mismatched')

    # Check missing values
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print('Data contains missing {} values'.format(n_missing))
        if na_action == 'drop':
            print('Dropping rows with missing values')
        elif na_action == 'ignore':
            print('Keeping missing values as it is')
        else:
            print('Not adding this data into master dataframe')
            check_passed = False

    return check_passed