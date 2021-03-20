from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from truenet.true_net import truenet_data_preprocessing
from skimage.transform import resize
import nibabel as nib

#=========================================================================================
# Truenet data postprocessing function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================

def resize_to_original_size(probs, testpathdicts, plane='axial'):
    '''
    :param probs: predicted 4d probability maps (N x H x W x Classes)
    :param testpathdicts: list of dictionaries containing test image datapaths
    :param plane: Acquisition plane
    :return: 3D probability maps with cropped dimensions.
    '''
    overall_prob = np.array([])
    st = 0    
    testpath = testpathdicts[0]
    flair_path = testpath['flair_path']
    data = nib.load(flair_path).get_data().astype(float)
    _,coords = truenet_data_preprocessing.tight_crop_data(data)
    if plane =='axial':
        probs_sub = probs[st:st+coords[5],:,:,:]
        prob_specific_sub = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[3]])
        for sli in range(probs_sub.shape[0]):
            prob_specific_sub[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[3]], preserve_range=True)
        overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
    elif plane == 'sagittal':
        probs_sub = probs[:128,:,:,:]
        probs_sub_resize = np.zeros([probs_sub.shape[0],coords[3],coords[5]])
        for sli in range(probs_sub.shape[0]):
            probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [coords[3], coords[5]], preserve_range=True)
        prob_specific_sub = probs_sub_resize.transpose(2,0,1)
        overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub 
    elif plane == 'coronal':
        probs_sub = probs[st:st+coords[3],:,:,:]
        probs_sub_resize = np.zeros([probs_sub.shape[0],128,coords[5]])
        for sli in range(probs_sub.shape[0]):
            probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [128, coords[5]], preserve_range=True)
        prob_specific_sub = probs_sub_resize.transpose(2,1,0)
        overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
    return overall_prob


def get_final_3dvolumes(volume3d,testpathdicts):
    '''
    :param volume3d: 3D probability maps
    :param testpathdicts: 3D probability maps in original dimensions
    :return:
    '''
    volume3d = np.tile(volume3d,(1,1,1,1))
    volume4d = volume3d.transpose(1,2,3,0)
    st = 0
    testpath = testpathdicts[0]
    flair_path = testpath['flair_path']
    data = nib.load(flair_path).get_data().astype(float)
    volume3d = 0 * data
    _,coords = truenet_data_preprocessing.tight_crop_data(data)
    row_cent = coords[1]//2 + coords[0]
    rowstart = np.amax([row_cent-64,0])
    rowend = np.amin([row_cent+64,data.shape[0]])
    colstart = coords[2]
    colend = coords[2] + coords[3]
    stackstart = coords[4]
    stackend = coords[4] + coords[5]
    data_sub = data[rowstart:rowend,colstart:colend,stackstart:stackend]
    required_stacks = volume4d[st:st+data_sub.shape[2],:data_sub.shape[0],:data_sub.shape[1],0].transpose(1,2,0)
    volume3d[rowstart:rowend,colstart:colend,stackstart:stackend] = required_stacks
    return volume3d


