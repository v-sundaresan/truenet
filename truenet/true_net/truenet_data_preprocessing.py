import numpy as np

#=========================================================================================
# Truenet data preprocessing function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================

def preprocess_data(data):
    '''
    Min-max intensity normalisation of data
    :param data: input data
    :return: normalised data
    '''
    return (2 * data / np.amax(np.reshape(data,[-1,1]))) - 1

def preprocess_data_gauss(data):
    '''
    Gaussian intensity normalisation of data
    :param data: input data
    :return: Gaussian normalised data
    '''
    brain1 = data > 0
    brain = brain1 > 0
    data = data - np.mean(data[brain])
    den = np.std(data[brain])
    if den == 0:
        den = 1
    data = data/den
    data[brain==0] = np.min(data)
    return data

def cut_zeros1d(im_array):
    '''
    Find the window for cropping the data closer to the brain
    :param im_array: input array
    :return: starting and end indices, and length of non-zero intensity values
    '''
    im_list = list(im_array > 0)
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:])-end_index
    return start_index, end_index, length

def tight_crop_data(img_data):
    '''
    Crop the data tighter to the brain
    :param img_data: input array
    :return: cropped image and the bounding box coordinates and dimensions.
    '''
    row_sum = np.sum(np.sum(img_data,axis=1),axis=1)
    col_sum = np.sum(np.sum(img_data,axis=0),axis=1)
    stack_sum = np.sum(np.sum(img_data,axis=1),axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen], [rsid, rlen, csid, clen, ssid, slen]
