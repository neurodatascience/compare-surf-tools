import nibabel as nib
from nibabel.freesurfer.io import read_morph_data
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
parser.add_argument('-l','--left',help='path for the left output csv for vertext-wise cortical thickness from freesurfer output')
parser.add_argument('-r','--right',help='path for the right output csv for vertext-wise cortical thickness from freesurfer output')
args = parser.parse_args()

surf_dir = args.path
file_l = args.left
file_r = args.right

subj_ID = surf_dir.split('/')[0]

l_surf_file = surf_dir + '/lh.thickness'
r_surf_file = surf_dir + '/rh.thickness'

try:
    l_surf = list(read_morph_data(l_surf_file))
    r_surf = list(read_morph_data(r_surf_file))

    print('subject {}, number of vertices L: {}, R: {}'.format(subj_ID, len(l_surf),len(r_surf)))

    write_csv([subj_ID] + l_surf, file_l)
    write_csv([subj_ID] + r_surf, file_r)

except:
    print('Unable to read thickness files')