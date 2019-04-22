# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 22 April 2019

import argparse
import os 
import sys

# input parser
parser = argparse.ArgumentParser(description='Process ROI-wise output from FS using aparcstats2table')
parser.add_argument('-s','--subjectdir',help='path for the file with subject list from freesurfer output')
parser.add_argument('-l','--listofsubjects',help='path for the file with subject list from freesurfer output')
parser.add_argument('-m','--meas',help='phenotypic measure (e.g. thickness, surface area etc.')
parser.add_argument('-p','--parc',help='cortical surface parcellation (e.g. a2009s,Glasseratlas')
parser.add_argument('-o','--output',help='path for the left and right output CSVs for ROI-wise cortical thickness from freesurfer output')

args = parser.parse_args()
subject_dir = args.subjectdir
subjectfile = args.listofsubjects
measure = args.meas
parc = args.parc

# output files
file_l = args.output + parc + '_' + measure + '_lh.csv'
file_r = args.output + parc + '_' + measure + '_rh.csv'

# freesurfer commands
cmd_l = 'aparcstats2table --hemi lh --subjectsfile {} --skip --meas {} --parc aparc.{} --tablefile {}'.format(subjectfile,measure,parc,file_l)
cmd_r = 'aparcstats2table --hemi rh --subjectsfile {} --skip --meas {} --parc aparc.{} --tablefile {}'.format(subjectfile,measure,parc,file_r)

# env var required by freesurfer
os.environ['SUBJECTS_DIR'] = subject_dir 

print('\ngenerating table for left hemisphere')
os.system(cmd_l)
print('\ngenerating table for right hemisphere')
os.system(cmd_r)