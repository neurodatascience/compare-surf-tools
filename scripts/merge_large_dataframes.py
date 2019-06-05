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
parser.add_argument('-n','--NumberOfSubjects', type=int, help='NumberOfSubjects in the combined CSV')
parser.add_argument('-b','--batch', type=int, help='batch size')
parser.add_argument('-o','--output',help='output csv for average thickness')

args = parser.parse_args()

vertex_file = args.ImageFeaturePath
demo_file = args.DemographicInfoPath
feat_col = args.feature
print('feature colums {}'.format(feat_col))
drop_condition = int(args.removeCols)
subx = args.NumberOfSubjects
batch_size = args.batch
out_csv = args.output

# Read demographics
Subject_id_col = 'SUB_ID'
demo_df_long = pd.read_csv(demo_file)
demo_df = demo_df_long[[Subject_id_col]+feat_col]

print('Shape of demo_df {}'.format(demo_df.shape))

# Create batches
n_iter = subx//batch_size
if subx%batch_size !=0:
    n_iter += 1


skip_rows = 0
demoMerged_csv = out_csv + 'demoMerged'
print('Splitting {} subjects into batches of {} giving {} iterations'.format(subx,batch_size,n_iter))  
for i in range(n_iter):
    if i == n_iter - 1:
        data_df = pd.read_csv(vertex_file, header=None, skiprows = skip_rows)
    else: 
        data_df = pd.read_csv(vertex_file, header=None, skiprows = skip_rows, nrows = batch_size)
    
    print('Reading rows {}:{}'.format(skip_rows,skip_rows+batch_size))
    mr_cols = list(range(data_df.shape[1]-1))
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
    print('Dropping columns with all 0s')
    
    # Need to read all rows and few columns to make sure every subject has 0s in that column (vertex)

    col_size = 10000
    non_zero_cols = []
    for start_idx in range(0,len(mr_cols),col_size):
        end_idx = start_idx + col_size
        print('Reading columns {}:{}'.format(start_idx,end_idx))
        col_subset = mr_cols[start_idx:end_idx]
        data_df =pd.read_csv(demoMerged_csv,usecols=col_subset)
        data_df = data_df.loc[:, (data_df != drop_condition).any(axis=0)]
        non_zero_cols = non_zero_cols + list(data_df.columns)
    
        del data_df

    non_zero_cols = [Subject_id_col] + non_zero_cols + feat_col
    print("Number of non-zero columns across all subjects: {}".format(len(non_zero_cols)))

    # One you have the list of nonzero columns you can re-read the csv by rows
    nonzero_csv = out_csv + '_nonzero.csv'
    print('Writing csv with dropped columns')
    skip_rows = 0
    print('n_iter {}'.format(n_iter))
    for i in range(n_iter):
        print('rows {}:{}'.format(skip_rows,skip_rows+batch_size))
        if i == 0:
            data_df = pd.read_csv(demoMerged_csv, nrows = batch_size)
            all_cols = data_df.columns
        elif i == n_iter - 1:
            print('i am here 0')
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

