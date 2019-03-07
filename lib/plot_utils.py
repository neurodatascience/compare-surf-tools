# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 March 2019

import os
import numpy as np
import nibabel as nib
from surfer import Brain
from mayavi import mlab

mlab.options.offscreen = True


def get_pysurfer_label_format(labels, aparc):
    ''' Check label format from freesurfer tables and convert it to pysurfer format
        Main differences are 1) prefix of 'L' and 'R' vs format char 'b' 2) '&' char instead of the word 'and'
    '''
    labels_std_L = []
    labels_std_R = []
    for label in labels:
        label_split = label.split('_',1)
        
        if aparc.lower() == 'glasser':
            label_std = label
        else:
            label_std = label_split[1]
        
        if label_split[0] == 'L':
            labels_std_L.append(label_std)
        elif label_split[0] == 'R':
            labels_std_R.append(label_std)
        else:
            print('unknown ROI label {}'.format(label))
            
    return labels_std_L, labels_std_R


def create_surface_plot(subject_id,hemi,surf,aparc,signific_rois,save_dir,title,view='lateral',signifcance_color=1):
    """
    Creates a pysurfer brain, overlays surface parcellation, and colormaps given ROIs 
    Used for plotting signficant ROIs
    """
    brain = Brain(subject_id, hemi, surf, background="white",title=title,views=view)


    aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                              subject_id, "label",
                              hemi + aparc)
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

    print('number of total vertices {} and ROIs {}'.format(len(labels),len(names)))
    
    # Convert names from bytes to strings
    names = [n.decode("utf-8").replace('-','_') for n in names]
          
    idx = []
    for roi in signific_rois:
        idx.append(names.index(roi))

    roi_value = np.zeros(len(names)) #np.random.randint(5, size=len(names)) #np.zeros(len(names))
    roi_value[idx] = np.arange(2,len(idx)+2) #signifcance_color

    print('number of significant rois {}'.format(len(signific_rois)))
          
    vtx_data = roi_value[labels]
          
    #Handle vertices that are not defined in the annotation.
    vtx_data[labels == -1] = -1
    unique, counts = np.unique(vtx_data, return_counts=True)
    print(dict(zip(unique, counts)))
    
    brain.add_data(vtx_data,colormap="icefire", alpha=.8, colorbar=True)
    save_path = '{}surf{}.png'.format(save_dir,title)
    brain.save_image(save_path)
    print('Image saved at {}'.format(save_path))
