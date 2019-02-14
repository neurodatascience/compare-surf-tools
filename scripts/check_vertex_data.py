# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 13 Feb 2019

import numpy as np
import pandas as pd
import argparse

# input parser
parser = argparse.ArgumentParser(description='Check vertex-wise output from NaNs in batches (avoid memory issues)')
parser.add_argument('-p','--path',help='path for the cortical thickness file from freesurfer output')
parser.add_argument('-n','--NumberOfSubjects',help='NumberOfSubjects in the combined CSV')
parser.add_argument('-b','--batch',help='batch size')
parser.add_argument('-o','--output',help='output csv for average thickness')
args = parser.parse_args()

vertex_file = args.path
subx = args.NumberOfSubjects
batch_size = args.batch
stat_csv = args.output

skip_rows = 0
n_iter = subx//batch_size + 1
missing_values = False

df_stats = pd.DataFrame(columns=['SubjID','mean_thickness'])

for i in range(n_iter):
    data = pd.read_csv(vertex_file, header=None, skiprows = skip_rows, nrows = batch_size)
    if data.isnull().values.any():
        missing_values = True
    else:
        df['SubjID'] = data.iloc[:,1]
        df['mean_thickness'] = data.iloc[:,1:].mean(axis=1)
        df_stats.append(df)

    del data
    skip_rows += batch_size

df_stats.to_csv(stat_csv)

