# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 22 April 2019

import argparse
import os 
import sys

# input parser
parser = argparse.ArgumentParser(description='Map custom parcellations (e.g. Glasser) on FS surfaces using mri_surf2surf and compute ROI stats (per subject) using mris_anatomical_stats')
parser.add_argument('-s','--subjectsdir',help='path for subjects dir from freesurfer output')
parser.add_argument('-p','--parcpath',help='path to the dir for left and right custom cortical surface parcellations')
parser.add_argument('-n','--name',help='name to the custom cortical surface parcellation (e.g. Glasser)')

args = parser.parse_args()

# subjects
subjects_dir = args.subjectsdir
subject_subdirs = os.listdir(subjects_dir)
n_subjects = len(subject_subdirs)
print('number of subjects found {}'.format(n_subjects))

# custome parc 
parcpath = args.parcpath
parc_name = args.name
print('using parc: {}'.format(parc_name))

sval_annot_lh = '{}lh.{}.annot'.format(parcpath, parc_name)
sval_annot_rh = '{}rh.{}.annot'.format(parcpath, parc_name)

tval_annot_lh = 'lh.{}.annot'.format(parc_name)
tval_annot_rh = 'rh.{}.annot'.format(parc_name)

tval_stats_lh = 'lh.{}.stats'.format(parc_name)
tval_stats_rh = 'rh.{}.stats'.format(parc_name)

# env var required by freesurfer
os.environ['SUBJECTS_DIR'] = subjects_dir 
FS_HOME = os.environ['FREESURFER_HOME']
fs_license = '{}/license.txt'.format(FS_HOME)

# Create sym link in subjects_dir
os.system('ln -s {}/subjects/fsaverage/ {}/fsaverage'.format(FS_HOME,subjects_dir))

# check for FS license file:
# /opt/freesurfer-6.0.0/license.txt
if not os.path.isfile(fs_license):
    print('Please copy FS license file here {}'.format(fs_license))

else:
    for subject in subject_subdirs:
        label_dir = '{}{}/label/'.format(subjects_dir,subject)
        stats_dir = '{}{}/stats/'.format(subjects_dir,subject)

        #------------ Map annotation from fsaverage space to the subject space (native) surfaces --------------#
        # mri_surf2surf --srcsubject fsaverage --trgsubject $s --hemi lh --sval-annot ../glasser/parcellation/lh.HCP-MMP1.annot --tval $SUBJECTS_DIR/${s}/label/lh.glasser.annot
        cmd_l = 'mri_surf2surf --srcsubject fsaverage --trgsubject {} --hemi lh --sval-annot {} --tval {}{}'.format(subject, sval_annot_lh, label_dir, tval_annot_lh)
        cmd_r = 'mri_surf2surf --srcsubject fsaverage --trgsubject {} --hemi rh --sval-annot {} --tval {}{}'.format(subject, sval_annot_rh, label_dir, tval_annot_rh)
        

        print('\ngenerating {} label for {} left hemisphere'.format(parc_name,subject))
        os.system(cmd_l)
        print('\ngenerating {} label for {} right hemisphere'.format(parc_name,subject))
        os.system(cmd_r)

        #------------ Compute stats for single subject parcellation --------------#
        # mris_anatomical_stats -a <subject>/label/lh.glasser.annot -f <subject>/stats/lh.aparc.Glasser.stats -b <subject> lh
        cmd_l = 'mris_anatomical_stats -a {}{} -f {}{} -b {} lh'.format(label_dir, tval_annot_lh, stats_dir, tval_stats_lh, subject)
        cmd_r = 'mris_anatomical_stats -a {}{} -f {}{} -b {} rh'.format(label_dir, tval_annot_rh, stats_dir, tval_stats_rh, subject)
        
        print('\ngenerating {} stats for {} left hemisphere'.format(parc_name,subject))
        os.system(cmd_l)
        print('\ngenerating {} stats for {} right hemisphere'.format(parc_name,subject))
        os.system(cmd_r)
        