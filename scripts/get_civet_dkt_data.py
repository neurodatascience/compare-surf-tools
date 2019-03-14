# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 12 March 2019

import sys
import os
import numpy as np
import pandas as pd
import itertools
import argparse

def getCIVETSubjectValues(atlas_df, subject_dir, subject_id, smoothing='30'):
    """ Parser for surfaces/sub-0050106_T1w_DKT_lobe_thickness_tlink_30mm_left.dat files from CIVET 2.1 output
        Uses DKT atlas. 
    """
    civet_subject_file = subject_dir + '/surfaces/sub-{}_T1w_DKT_lobe_thickness_tlink_{}mm_{}.dat'
    civet_subject_both_hemi = pd.DataFrame()
    for hemi in ['left','right']:
        try:
            civet_subject = pd.read_csv(civet_subject_file.format(subject_id,smoothing,hemi),header=1,
                                        delim_whitespace=True)
            civet_subject = civet_subject[['#','Label']]
            civet_subject = civet_subject.rename(columns={'#':'roi_id','Label':subject_id})
            civet_subject = civet_subject[civet_subject['roi_id']!='Total']
            civet_subject_both_hemi = civet_subject_both_hemi.append(civet_subject)
            civet_subject_both_hemi['roi_id'] = civet_subject_both_hemi['roi_id'].astype('int') 
            civet_subject_both_hemi = pd.merge(atlas_df,civet_subject_both_hemi,on='roi_id')
            
            civet_subject_both_hemi = civet_subject_both_hemi[['roi_name',subject_id]].set_index('roi_name').T
            civet_subject_both_hemi = civet_subject_both_hemi.rename_axis('SubjID').rename_axis(None, 1)
            
        except FileNotFoundError:
            print("File doesn't exist {}".format(civet_subject_file))
        else:
            break

    return civet_subject_both_hemi


# input parser
parser = argparse.ArgumentParser(description='Read DKT output from CIVET2.1 and create group table')
parser.add_argument('-p','--path',help='path for subject dir from CIVET output')
parser.add_argument('-s','--smoothing',help='smoothing kernel (20,30,40mm)')
parser.add_argument('-n','--nameprefix',help='naming prefix for subjects used by civet')
parser.add_argument('-a','--atlas',help='atlas path for (AAL or DKT)') #DKT only atm
parser.add_argument('-o','--output',help='output csv for average thickness')

args = parser.parse_args()
civet_out_dir = args.path
name_prefix = args.nameprefix
atlas_file = args.atlas
smoothing = args.smoothing
save_path = args.output

#civet_subject_dir = civet_test_dir + 'test_subjects/sub-{}_T1w/'
#civet_dkt_atlas_file = civet_test_dir + 'DKT/DKTatlas40.labels'
#subject_ids = ['0050106','0050106']

all_dirs = next(os.walk(civet_out_dir))[1]
sub_dirs = [d for d in all_dirs if d.startswith(name_prefix)]

print('Lookig for subjects in {} ...'.format(civet_out_dir))
print('Number of subject directories found {}'.format(len(sub_dirs)))

# Read atlas file
print('Reading atlas from {}'.format(atlas_file))
civet_atlas = pd.read_csv(atlas_file,header=None,delim_whitespace=True)
civet_atlas.columns = ['roi_id','roi_name']
civet_atlas['roi_id'] = civet_atlas['roi_id'].astype('int')

civet_master_df = pd.DataFrame()

for sub_dir in sub_dirs:
    sub_dir_path = '{}/{}'.format(civet_out_dir,sub_dir)
    subject_id = sub_dir.split('-',1)[1].split('_',1)[0]
    civet_subject_df = getCIVETSubjectValues(civet_atlas, sub_dir_path, subject_id, smoothing)
    civet_master_df = civet_master_df.append(civet_subject_df)

civet_master_df.to_csv(save_path)
print('Saving CIVET subject-group data to {}'.format(save_path))

