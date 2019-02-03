# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd
from data_handling import *

# Data paths
proj_dir = '/Users/nikhil/code/git_repos/compare-surf-tools/'
data_dir = proj_dir + 'data/'
ants_file = 'ABIDE_ants_thickness_data.csv'
fs53_file = 'ABIDE_fs5.3_thickness.csv'
fs51_file = 'cortical_fs5.1_measuresenigma_thickavg.csv' 
fs60_lh_file = 'aparc_lh_thickness_table.txt'
fs60_rh_file = 'aparc_rh_thickness_table.txt'


# test_1: stdize data
test_name = 'test_1: stdize data'
print('\n ------------- Running {} -------------'.format(test_name))

subject_ID_col = 'SubjID'
# ANTs
ants_data = pd.read_csv(data_dir + ants_file, header=2)
print('shape of ants data {}'.format(ants_data.shape))
ants_data_std = standardize_ants_data(ants_data, subject_ID_col)
print('shape of stdized ants data {}'.format(ants_data_std.shape))
print(list(ants_data_std.columns)[:5])
print('')

# FS
fs53_data = pd.read_csv(data_dir + fs53_file)
print('shape of fs51 data {}'.format(fs53_data.shape))
fs53_data_std = standardize_fs_data(fs53_data, subject_ID_col)
print('shape of stdized fs53 data {}'.format(fs53_data_std.shape))
print(list(fs53_data_std.columns[:5]))
print('')

fs51_data = pd.read_csv(data_dir + fs51_file)
print('shape of fs51 data {}'.format(fs51_data.shape))
fs51_data_std = standardize_fs_data(fs51_data, subject_ID_col)
print('shape of stdized fs51 data {}'.format(fs51_data_std.shape))
print(list(fs51_data_std.columns[:5]))
print('')

fs60_lh_data = pd.read_csv(data_dir + fs60_lh_file, delim_whitespace=True)
fs60_rh_data = pd.read_csv(data_dir + fs60_rh_file, delim_whitespace=True)
print('shape of fs60 data l: {}, r: {}'.format(fs60_lh_data.shape,fs60_rh_data.shape))
fs60_data_std = standardize_fs60_data(fs60_lh_data, fs60_rh_data, subject_ID_col)
print('shape of stdized fs51 data {}'.format(fs60_data_std.shape))
print(list(fs60_data_std.columns[:5]))

# test_2: create master df
test_name = 'test_2: create master df'
print('\n ------------- Running {} -------------'.format(test_name))

data_dict = {'ants' : ants_data_std,
            'fs60' : fs60_data_std,
            'fs53' : fs53_data_std,
            'fs51' : fs51_data_std}

na_action = 'drop' # options: ignore, drop; anything else will not use the dataframe for analysis. 
test = combine_processed_data(data_dict, subject_ID_col, na_action)