# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 13 Feb 2019

import nibabel as nib
from nibabel.freesurfer.io import read_morph_data
from nibabel.freesurfer.mghformat import load
import numpy as np

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
parser.add_argument('-p','--path',help='path for the cortical thickness file from freesurfer output')
parser.add_argument('-s','--suffix',help='suffix of the cortical thickness file from freesurfer output')
parser.add_argument('-o','--output',help='path for the left and right output CSVs for vertext-wise cortical thickness from freesurfer output')

args = parser.parse_args()

surf_dir = args.path
suffix = args.suffix
file_l = args.output + '_' + suffix + '_lh.csv'
file_r = args.output + '_' + suffix + '_rh.csv'

subj_ID = surf_dir.rsplit('/',2)[1]

l_surf_file = surf_dir + '/lh.thickness' + suffix
r_surf_file = surf_dir + '/rh.thickness' + suffix

try:
    # l_surf = list(read_morph_data(l_surf_file)) # Only works for .thickness files
    # r_surf = list(read_morph_data(r_surf_file))
    l_surf = list(np.squeeze(load(l_surf_file).get_data()))
    r_surf = list(np.squeeze(load(r_surf_file).get_data()))
    
    print('subject {}, number of vertices L: {}, R: {}'.format(subj_ID, len(l_surf),len(r_surf)))

    write_csv([subj_ID] + l_surf, file_l)
    write_csv([subj_ID] + r_surf, file_r)

except:
    print('Unable to read thickness files')