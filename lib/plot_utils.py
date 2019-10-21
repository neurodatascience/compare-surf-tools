# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 March 2019

import os
import numpy as np
import nibabel as nib
from surfer import Brain
from mayavi import mlab
from scipy.spatial import distance

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

mlab.options.offscreen = True


def get_pysurfer_label_format(labels, aparc, betas=None):
    ''' Check label format from freesurfer tables and convert it to pysurfer format
        Main differences are 1) prefix of 'L' and 'R' vs format char 'b' 2) '&' char instead of the word 'and'
    '''
    labels_std_L = []
    labels_std_R = []
    betas_L = []
    betas_R = []
    for i, label in enumerate(labels):
        label_split = label.split('_',1)
        
        if aparc.lower() == 'glasser':
            label_std = label
        else:
            label_std = label_split[1]
        
        if label_split[0] == 'L':
            labels_std_L.append(label_std)
            betas_L.append(betas[i])
        elif label_split[0] == 'R':
            labels_std_R.append(label_std)
            betas_R.append(betas[i])
        else:
            print('unknown ROI label {}'.format(label))
            
    return labels_std_L, labels_std_R, betas_L, betas_R


def create_surface_plot(common_space,hemi,surf,aparc_file,signific_rois,save_dir,title,view,signifcance_color=[],plot_style={}):
    """
    Creates a pysurfer brain, overlays surface parcellation, and colormaps given ROIs 
    Used for plotting signficant ROIs
    If significance color (effect size / betas) are provided (in the same order as significant ROIs) then uses it as a colormap
    """
    brain = Brain(common_space, hemi, surf, background="white",title=title,views=view)

    if len(signific_rois) > 0:
        aparc_file_path = os.path.join(os.environ["SUBJECTS_DIR"],
                                common_space, "label",
                                hemi + aparc_file)
        labels, ctab, names = nib.freesurfer.read_annot(aparc_file_path)

        print('number of total vertices {} and ROIs {}'.format(len(labels),len(names)))
        
        # Convert names from bytes to strings
        names = [n.decode("utf-8").replace('-','_') for n in names]
            
        idx = []
        for roi in signific_rois:
            idx.append(names.index(roi))

        #print('idx-betas \n{} \n{}'.format(list(zip(idx,signifcance_color)), np.array(names)[idx]))
        roi_value = np.zeros(len(names)) #np.random.randint(5, size=len(names)) #np.zeros(len(names))
        if len(signifcance_color) == 0:
            roi_value[idx] = np.arange(2,len(idx)+2) #random signifcance_color
        else:
            roi_value[idx] = signifcance_color
            print('Using betas as colormap')

        print('number of significant rois {}'.format(len(signific_rois)))
            
        vtx_data = roi_value[labels]
            
        #Handle vertices that are not defined in the annotation.
        vtx_data[labels == -1] = 0

        unique, counts = np.unique(vtx_data, return_counts=True)
        print('atlas: {}, signficant roi count: {}'.format(aparc_file, dict(zip(unique, counts))))

        #plot style and aesthetic
        if 'colormap' in plot_style.keys():
            colormap = plot_style['colormap']
        else:
            colormap = 'icefire'

        if 'center' in plot_style.keys():
            c_center = plot_style['center']
            brain.add_data(vtx_data,colormap=colormap, alpha=.8, colorbar=True, center=c_center)
        elif 'range' in plot_style.keys():
            c_min = plot_style['range'][0]
            c_max = plot_style['range'][1]
            brain.add_data(vtx_data,colormap=colormap, alpha=.8, colorbar=True, min=c_min,max=c_max)
        else:
            brain.add_data(vtx_data,colormap=colormap, alpha=.8, colorbar=True)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = '{}surf{}.png'.format(save_dir,title)
    brain.save_image(save_path)
    print('Image saved at {}'.format(save_path))


def createImageMontage(img_dir,thumb_size=200,font_size=24,num_img_views=4,transpose=False):
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
        #montage_size = (n_row*pane_size,n_col*pane_size)
        if transpose:
            montage_size = (n_row*pane_size,n_col*pane_size)
        else:
            montage_size = (n_col*pane_size,n_row*pane_size)
            
        montage_im = Image.new('RGB', montage_size,color=(255,255,255,0))

        print('montage size {}'.format(montage_size))
        
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
                if transpose:
                    montage_im.paste(im, (j,i))
                else:    
                    montage_im.paste(im, (i,j))

                imx+=1

    else:
        print("Not all pipeline vaiations (e.g. tools,atlases) have same number of images.")
        montage_im = None
        
    return montage_im


def createSingleImageMontage(img_dir,thumb_size=200,font_size=24,num_img_views=4,transpose=False):
    """
    Creates montage with 4 views for a single image.
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

    n_row = 2
    if len(imagex)%n_row == 0: #Make sure each atlas / tool have same number (4x) of images 
        n_col = len(imagex)//n_row

        #creates a new empty image, RGB mode
        #montage_size = (n_row*pane_size,n_col*pane_size)
        if transpose:
            montage_size = (n_row*pane_size,n_col*pane_size)
        else:
            montage_size = (n_col*pane_size,n_row*pane_size)
            
        montage_im = Image.new('RGB', montage_size,color=(255,255,255,0))

        print('montage size {}'.format(montage_size))
        
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
                if transpose:
                    montage_im.paste(im, (j,i))
                else:    
                    montage_im.paste(im, (i,j))

                imx+=1

    else:
        print("Not all pipeline vaiations (e.g. tools,atlases) have same number of images.")
        montage_im = None
        
    return montage_im


def get_nbrs(coords,vertex_idx):
    seeds = coords[vertex_idx]
    dist_array = distance.cdist(seeds, coords, 'euclidean')

    aug_vertex_idx =[]
    n_nbrs = 50
    for s in range(len(seeds)):
        closest_v = dist_array[s,:].argsort()[:n_nbrs]
        aug_vertex_idx.append(closest_v)

    return np.hstack((aug_vertex_idx))

def plot_surface_vertices(common_space,morph_data,vertex_idx,aug_data,hemi,surf,view,cmap,save_path):
    b = Brain(common_space, hemi, surf, background="white",views=view)
    x, y, z = b.geo[hemi].coords.T
    coords = np.array([x,y,z]).T
    print('Number of vertices to be plotted {}'.format(len(vertex_idx)))
    if aug_data:
        aug_vertex_idx = get_nbrs(coords,vertex_idx)
        print('Number of vertices to be plotted after augmentation {}'.format(len(aug_vertex_idx)))
        morph_data[aug_vertex_idx] = 1 
    else:
        morph_data[vertex_idx] = 1 
    
    print(np.sum(morph_data))
    b.add_data(morph_data,colormap=cmap, alpha=.9, colorbar=True)    
    print(save_path)
    b.save_image(save_path)