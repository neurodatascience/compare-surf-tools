# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 24 May 2019

import numpy as np
import pandas as pd
import argparse

# input parser
parser = argparse.ArgumentParser(description='Merge demographic info onto high-dim image feature csv')
parser.add_argument('-i','--ImageFeaturePath',help='path for the imaging feature file (high-dim)')
parser.add_argument('-d','--DemographicInfoPath',help='path for the demographic file')
parser.add_argument('-f','--feature',nargs='+', help='feature column from the demographic file')

parser.add_argument('-r','--removeCols',help='drop columns with this condition')
parser.add_argument('-n','--NumberOfSubjects', type=int, help='NumberOfSubjects in the combined CSV')
parser.add_argument('-b','--batch', type=int, help='batch size')
parser.add_argument('-o','--output',help='output csv for average thickness')

args = parser.parse_args()

vertex_file = args.ImageFeaturePath
demo_file = args.DemographicInfoPath
feat_col = args.feature
drop_condition = args.removeCols
subx = args.NumberOfSubjects
batch_size = args.batch
out_csv = args.output

# Read demographics
Subject_id_col = 'SUB_ID'
demo_df_long = pd.read_csv(demo_file)
demo_df = demo_df_long[[Subject_id_col,feat_col]]

print('Shape of demo_df {}'.format(demo_df.shape))

# Create batches
n_iter = subx//batch_size
if subx%batch_size !=0:
    n_iter += 1

skip_rows = 0

print('Splitting {} subjects into batches of {} giving {} iterations'.format(subx,batch_size,n_iter))  
for i in range(n_iter):
    if i == n_iter - 1:
        data_df = pd.read_csv(vertex_file, header=None, skiprows = skip_rows)
    else: 
        data_df = pd.read_csv(vertex_file, header=None, skiprows = skip_rows, nrows = batch_size)
    
    if drop_condition is not None:
        print('Dropping columns with all 0s')
        data_df = data_df.loc[:, (data_df != drop_condition).any(axis=0)]

    print('rows {}:{}'.format(skip_rows,skip_rows+len(data_df)))
    data_df.columns = [Subject_id_col] + list(range(data_df.shape[1]-1))
    data_df = pd.merge(data_df,demo_df,on=Subject_id_col,how='left')
    
    with open(out_csv, 'a') as f:
        if i==0:
            data_df.to_csv(f, header=True)
        else:
            data_df.to_csv(f, header=False)
    
    del data_df
    skip_rows += batch_size

print('Saving merged CSV here: {}'.format(out_csv))