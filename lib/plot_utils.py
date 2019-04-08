# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 March 2019

import os
import numpy as np
import nibabel as nib
from surfer import Brain
from mayavi import mlab

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

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

    if len(signific_rois) > 0:
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
        print('atlas: {}, signficant roi count: {}'.format(aparc, dict(zip(unique, counts))))
        
        brain.add_data(vtx_data,colormap="icefire", alpha=.8, colorbar=True)
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = '{}surf{}.png'.format(save_dir,title)
    brain.save_image(save_path)
    print('Image saved at {}'.format(save_path))


def createImageMontage(img_dir,thumb_size=200,font_size=24,num_img_views=4):
    """
    Creates montages of all images in a given dir. 
    Assumes all pipeline variations (tool / atlas) have same numbre of images (views).
    Views used: lh,rh,lat,med
    """
    
    # Montage properties (size and text)
    thumb_size = thumb_size
    pane_size = thumb_size + thumb_size//10 #Add 10%padding
    font = ImageFont.truetype("arial.ttf", font_size)
    font_color = (0,0,0)
    text_loc = (thumb_size//10,thumb_size//10)

    # Get list of all images in the directory 
    print("Reading images from {}".format(img_dir))
    imagex =  sorted(next(os.walk(img_dir))[2])
    print("Found {} images".format(len(imagex)))

    n_row = num_img_views #lh,rh,med,lat
    if len(imagex)%n_row == 0: #Make sure each atlas / tool have same number (4x) of images 
        n_col = len(imagex)//n_row

        #creates a new empty image, RGB mode
        montage_im = Image.new('RGB', (n_col*pane_size,n_row*pane_size),color=(255,255,255,0))

        imx = 0
        
        #Iterate through a grid 
        for i in range(0,(n_col)*pane_size, pane_size):
            for j in range(0,(n_row)*pane_size, pane_size):                                   
                #opn image
                img_file = imagex[imx]
                im = Image.open(img_dir + img_file)

                #insert title
                draw = ImageDraw.Draw(im)
                draw.text(text_loc,img_file,font_color,font=font)

                #resize
                im.thumbnail((thumb_size,thumb_size))

                #paste the image at location i,j:
                montage_im.paste(im, (i,j))

                imx+=1

    else:
        print("Not all pipeline vaiations (e.g. tools,atlases) have same number of images.")
        montage_im = None
        
    return montage_im