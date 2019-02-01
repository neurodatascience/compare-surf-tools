# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd
from data_handling import *

proj_dir = '/Users/nikhil/code/git_repos/compare-surf-tools/'
data_dir = proj_dir + 'data/'

# test_1: Data load
csv_dict = {'ants' : data_dir + 'ABIDE_ants_thickness_data.csv', 
            'fs' : data_dir + 'ABIDE_fs5.3_thickness.csv'}

col_list = pd.read_csv(data_dir + 'ABIDE_ants_thickness_data.csv').columns

# Run test_1
na_action = 'drop' # options: ignore, drop; anything else will not use the dataframe for analysis. 
test = load_processed_data(csv_dict,col_list,na_action)