# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 13 Feb 2019

import nibabel as nib
from nibabel.freesurfer.io import read_morph_data
from nibabel.freesurfer.mghformat import load
import numpy as np
import os
import argparse
import csv

# Local defs
def write_csv(v_list,filename):
    with open(filename, 'a') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerow(v_list)

# input parser
#f = '/home/nikhil/projects/rpp-aevans-ab/flatcbraindir/run1_MaxMun_a_0051362.out/sub-0051362/surf/lh.thickness'
parser = argparse.ArgumentParser(description='Process vertex-wise output from FS')
parser.add_argument('-s','--subjectsdir',help='path for subjects dir from freesurfer output')
parser.add_argument('-k','--kernel',help='smoothing factor tag of the cortical thickness file from freesurfer output')
parser.add_argument('-o','--output',help='path for the left and right output CSVs for vertext-wise cortical thickness from freesurfer output')

args = parser.parse_args()

subjects_dir = args.subjectsdir
subject_subdirs = os.listdir(subjects_dir)
n_subjects = len(subject_subdirs)
print('number of subjects found {}'.format(n_subjects))

for subject_dir in subject_subdirs: 
    surf_dir = subjects_dir + subject_dir + '/surf'
    suffix = args.kernel
    l_surf_file = surf_dir + '/lh.thickness' + suffix
    r_surf_file = surf_dir + '/rh.thickness' + suffix

    # output CSVs
    l_out_file = args.output + suffix + '_lh.csv'
    r_out_file = args.output + suffix + '_rh.csv'

    subj_ID = surf_dir.rsplit('/',2)[1]

    try:
        # l_surf = list(read_morph_data(l_surf_file)) # Only works for .thickness files
        # r_surf = list(read_morph_data(r_surf_file))
        l_surf = list(np.squeeze(load(l_surf_file).get_data()))
        r_surf = list(np.squeeze(load(r_surf_file).get_data()))
        
        #print('subject {}, number of vertices L: {}, R: {}'.format(subj_ID, len(l_surf),len(r_surf)))

        write_csv([subj_ID] + l_surf, l_out_file)
        write_csv([subj_ID] + r_surf, r_out_file)

    except:
        print('Unable to read thickness files')

print('vertex-wise CSVs saved to {}_[lh,rh].csv'.format(args.output))