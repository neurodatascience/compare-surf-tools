# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 12 March 2019

import sys
import numpy as np
import pandas as pd
import itertools


def getCIVETSubjectValues(atlas_df, subject_dir, subject_id, smoothing):
    """ Parser for surfaces/sub-0050106_T1w_DKT_lobe_thickness_tlink_30mm_left.dat files from CIVET 2.1 output
        Uses DKT atlas. 
    """
    civet_subject_file = subject_dir + 'surfaces/sub-{}_T1w_DKT_lobe_thickness_tlink_{}_{}.dat'
    civet_subject_both_hemi = pd.DataFrame()
    for hemi in ['left','right']:
        civet_subject = pd.read_csv(civet_subject_file.format(subject_id,subject_id,smoothing,hemi),header=1,
                                    delim_whitespace=True)
        civet_subject = civet_subject[['#','Label']]
        civet_subject = civet_subject.rename(columns={'#':'roi_id','Label':subject_id})
        civet_subject = civet_subject[civet_subject['roi_id']!='Total']
        civet_subject_both_hemi = civet_subject_both_hemi.append(civet_subject)

    civet_subject_both_hemi['roi_id'] = civet_subject_both_hemi['roi_id'].astype('int') 
    civet_subject_both_hemi = pd.merge(atlas_df,civet_subject_both_hemi,on='roi_id')
    
    civet_subject_both_hemi = civet_subject_both_hemi[['roi_name',subject_id]].set_index('roi_name').T
    civet_subject_both_hemi = civet_subject_both_hemi.rename_axis('SubjID').rename_axis(None, 1)
    
    return civet_subject_both_hemi



# TODO
# Replace below with argparse

civet_test_dir = '/home/nikhil/projects/CT_reproduce/data/civet_test_dir/'
civet_subject_dir = civet_test_dir + 'test_subjects/sub-{}_T1w/'
civet_dkt_atlas_file = civet_test_dir + 'DKT/DKTatlas40.labels'

civet_dkt_atlas = pd.read_csv(civet_dkt_atlas_file,header=None,delim_whitespace=True)
civet_dkt_atlas.columns = ['roi_id','roi_name']
civet_dkt_atlas['roi_id'] = civet_dkt_atlas['roi_id'].astype('int')

subject_ids = ['0050106','0050106']
smoothing = '30mm'

civet_master_df = pd.DataFrame()

for subject_id in subject_ids:
    civet_subject_df = getCIVETSubjectValues(civet_dkt_atlas, civet_subject_dir, subject_id, smoothing)
    civet_master_df = civet_master_df.append(civet_subject_df)

civet_master_df.to_csv(save_path)
print('Saving CIVET subject-group data to {}'.format(save_path))

