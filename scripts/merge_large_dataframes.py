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
parser.add_argument('-f','--feature',nargs='+', help='feature column from the demographic file',type=str)
parser.add_argument('-r','--removeCols',help='drop columns with this condition - needs to be a number')
parser.add_argument('-n','--NameOfSubjectColumn', type=str, help='Column name for subject ID')
parser.add_argument('-b','--batch', type=int, help='batch size')
parser.add_argument('-c','--columnHeader', type=bool, help='batch size')
parser.add_argument('-o','--output',help='output csv for average thickness')

args = parser.parse_args()

vertex_file = args.ImageFeaturePath
demo_file = args.DemographicInfoPath
feat_col = args.feature
drop_condition = args.removeCols
Subject_id_col = args.NameOfSubjectColumn
batch_size = args.batch
out_csv = args.output
header = args.columnHeader
demoMerged_csv = out_csv + 'demoMerged'
nonzero_csv = out_csv + '_nonzero.csv'

# Get number of subjects and vertecies
tmp_df = pd.read_csv(vertex_file, header=header, nrows=1)
n_col = tmp_df.shape[1]-1 # all columns except  Subject ID
del tmp_df
tmp_df = pd.read_csv(vertex_file, header=header, usecols=[0]) #first column is Subject ID
n_sub = tmp_df.shape[0]

print('Number of subjects {}, number of vertices {}'.format(n_sub, n_col))

mr_cols = list(range(n_col))
col_size = 10000 #number of colums to read at a time to avoid mem issues

# Create batches
n_iter = n_sub//batch_size
if n_sub%batch_size !=0:
    n_iter += 1

if demo_file is not None: 
    print('Merging demographic info')
    # Read demographics
    demo_df_long = pd.read_csv(demo_file)
    demo_df = demo_df_long[[Subject_id_col]+feat_col]
    print('Shape of demo_df {}'.format(demo_df.shape))
    print('Using these demographic columns for merge {}'.format(feat_col))

    skip_rows = 0
    print('Splitting {} subjects into batches of {} giving {} iterations'.format(n_sub,batch_size,n_iter))  
    for i in range(n_iter):
        if i == n_iter - 1:
            data_df = pd.read_csv(vertex_file, header=header, skiprows = skip_rows)
        else: 
            data_df = pd.read_csv(vertex_file, header=header, skiprows = skip_rows, nrows = batch_size)
        
        print('Reading rows {}:{}'.format(skip_rows,skip_rows+batch_size))
        data_df.columns = [Subject_id_col] + mr_cols
        data_df = pd.merge(data_df,demo_df,on=Subject_id_col,how='left')

        with open(demoMerged_csv, 'a') as f:
            if i==0:
                data_df.to_csv(f, header=True)
            else:
                data_df.to_csv(f, header=False)
        
        del data_df
        skip_rows += batch_size

if drop_condition is not None:
    drop_condition = int(drop_condition)
    print('Dropping columns with all {}',format(drop_condition))
    # Need to read all rows and few columns to make sure every subject has 0s in that column (vertex)
    
    n_iter_col = n_col//col_size
    if n_col%col_size !=0:
        n_iter_col += 1

    non_zero_cols = []
    start_col = 0
    mr_cols = list(map(str, mr_cols)) # Column names are strings, int dtype will use it as a (wrong) col index
    for i in range(n_iter_col):
        end_col = start_col + col_size
        print('Reading columns {}:{}'.format(start_col,end_col))
        col_subset =  mr_cols[start_col:end_col]
        data_df =pd.read_csv(demoMerged_csv,usecols=col_subset)
        data_df = data_df.loc[:, (data_df != drop_condition).any(axis=0)]
        non_zero_cols = non_zero_cols + list(data_df.columns)
        start_col += col_size
        del data_df

    non_zero_cols = [Subject_id_col] + non_zero_cols + feat_col
    print("Number of non-zero columns across all subjects: {}".format(len(non_zero_cols)))
    print(non_zero_cols[:5])

    # One you have the list of nonzero columns you can re-read the csv by rows
    print('Writing csv with dropped columns')
    skip_rows = 0
    print('n_iter {}'.format(n_iter))
    for i in range(n_iter):
        print('rows {}:{}'.format(skip_rows,skip_rows+batch_size))
        if i == 0:
            data_df = pd.read_csv(demoMerged_csv, nrows = batch_size)
            all_cols = data_df.columns
        elif i == n_iter - 1:
            data_df = pd.read_csv(demoMerged_csv, header=None,  skiprows = skip_rows+1)
        else: 
            data_df = pd.read_csv(demoMerged_csv, header=None,  skiprows = skip_rows+1, nrows=batch_size)
        
        data_df.columns = all_cols
        data_df = data_df[non_zero_cols]
        print('df shape after dropping zeros {}'.format(data_df.shape))
        
        
        with open(nonzero_csv, 'a') as f:
            if i==0:
                data_df.to_csv(f, header=True)
            else:
                data_df.to_csv(f, header=False)
    
        del data_df
        skip_rows += batch_size

print('Saving merged CSV here: {}'.format(nonzero_csv))