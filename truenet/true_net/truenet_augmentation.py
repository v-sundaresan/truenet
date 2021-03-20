from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise

#=========================================================================================
# Truenet augmentations function
# Vaanathi Sundaresan
# 11-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define transformations with distance maps
##########################################################################################

def translate_distmaps2d(image, imaget1, label, gmdist, ventdist):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual lesion mask
    :param gmdist: GM distance map
    :param ventdist: Ventrile distance map
    :return: translated FLAIR, T1, manual lesion mask, GM and ventricle distance maps
    '''
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_imt1 = shift(imaget1, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety), order=order, mode='nearest')
    translated_gmdist = shift(gmdist,(offsetx, offsety), order=order, mode='nearest')
    translated_ventdist = shift(ventdist,(offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_imt1, translated_label, translated_gmdist, translated_ventdist

def rotate_distmaps2d(image, imaget1, label,gmdist, ventdist):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual lesion mask
    :param gmdist: GM distance map
    :param ventdist: Ventricle distance map
    :return: rotated FLAIR, T1, manual lesion mask, GM and ventricle distance maps
    '''
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_imgt1 = rotate(imaget1, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_imgt1, new_lab, gmdist, ventdist


def blur_distmaps2d(image, imaget1, label, gmdist, ventdist):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual lesion mask
    :param gmdist: GM distance map
    :param ventdist: Ventricle distance map
    :return: blurred FLAIR and T1 images, input manual mask, GM and ventricle distance maps
    '''
    sigma = random.uniform(0.1,0.2)
    new_img = gaussian_filter(image, sigma)
    new_imgt1 = gaussian_filter(imaget1, sigma)
    return new_img, new_imgt1, label, gmdist, ventdist

def add_noise_to_it_distmaps2d(image,imaget1,label,gmdist,ventdist):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual lesion mask
    :param gmdist: GM distance map
    :param ventdist: Ventricle distance map
    :return: noise injected FLAIR and T1 images, input manual mask, GM and Ventricle distance maps
    '''
    new_img = random_noise(image,clip=False)
    new_imgt1 = random_noise(imaget1,clip=False)
    new_lab = label
    return new_img, new_imgt1, new_lab, gmdist, ventdist

##########################################################################################
# Define transformations without distance maps
##########################################################################################

def translate_it2d(image, imaget1, label):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: Manual mask
    :return: translated FLAIR, T1 and manual mask
    '''
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_imt1 = shift(imaget1, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_imt1, translated_label

def rotate_it2d(image, imaget1, label):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: Manual mask
    :return: rotated FLAIR, T1 and manual mask
    '''
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_imgt1 = rotate(imaget1, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_imgt1, new_lab


def blur_it2d(image, imaget1, label):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual mask
    :return: blurred FLAIR and T1 images, input manual mask
    '''
    sigma = random.uniform(0.1,0.2)
    new_img = gaussian_filter(image, sigma)
    new_imgt1 = gaussian_filter(imaget1, sigma)
    return new_img, new_imgt1, label

def add_noise_to_it2d(image,imaget1,label):
    '''
    :param image: FLAIR
    :param imaget1: T1
    :param label: manual mask
    :return: noise injected FLAIR and T1 images, input manual mask
    '''
    new_img = random_noise(image,clip=False)
    new_imgt1 = random_noise(imaget1,clip=False)
    new_lab = label
    return new_img, new_imgt1, new_lab

##########################################################################################
# Define augmentation main function with distance maps
##########################################################################################

def augment_distmaps(image_to_transform, imaget1, label, gmdist, ventdist):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image_to_transform, imaget1: input images as arrays
    :param label: label for input image to also transform
    :param gmdist: label for GM distance image to also transform
    :param ventdist: label for ventricle distance image to also transform
    :return: transformed image, transformed label and transformed distance maps
    """
    if len(image_to_transform.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise_to_it_distmaps2d, 'translate': translate_distmaps2d,
                                     'rotate': rotate_distmaps2d, 'blur': blur_distmaps2d}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_imaget1 = None
        transformed_label = None
        transformed_gmdist = None
        transformed_ventdist = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist = available_transformations[key](image_to_transform, imaget1, label, gmdist, ventdist)
            num_transformations += 1
        return transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')

##########################################################################################
# Define transformations without distance maps
##########################################################################################

def augment(image_to_transform, imaget1, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image_to_transform, imaget1: input images as arrays
    :param label: optional: label for input image to also transform
    :return: transformed image and transformed label
    """
    if len(image_to_transform.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise_to_it2d, 'translate': translate_it2d,
                                     'rotate': rotate_it2d, 'blur': blur_it2d}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_imaget1 = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_imaget1, transformed_label = available_transformations[key](image_to_transform, imaget1, label)
            num_transformations += 1
        return transformed_image, transformed_imaget1, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')
