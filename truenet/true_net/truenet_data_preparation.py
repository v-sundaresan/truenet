import numpy as np
from truenet.true_net import (truenet_augmentation, truenet_data_preprocessing)
from skimage.transform import resize
import nibabel as nib

#=========================================================================================
# Truenet data preparation function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================

def create_data_array(names, is_weighted=True, plane='axial'):
    '''
    Create the input stack of 2D slices reshaped to required dimensions
    :param names: list of dictionaries containing filepaths
    :param is_weighted: bool, whether to use spatial weighting in loss function
    :param plane: acquisition plane
    :return: dictionary of input arrays
    '''
    data = np.array([])
    data_t1 = np.array([])
    labels = np.array([])
    GM_distance = np.array([])
    ventdistmap = np.array([])

    for i in range(len(names)):
        array_loaded = load_and_crop_data(names[i], weighted=is_weighted)
        data_sub1 = array_loaded['data_cropped']
        data_t1_sub1 = array_loaded['data_t1_cropped']
        labels_sub1 = array_loaded['label_cropped']
        if is_weighted:
            GM_distance_sub1 = array_loaded['gmdist_cropped']
            ventdistmap_sub1 = array_loaded['ventdist_cropped']

        if plane == 'axial':
            if data_sub1 is not None:
                data_sub1 = data_sub1.transpose(2, 0, 1)
                data = np.concatenate(
                    (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192], preserve_range=True)),
                    axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192],
                                                     preserve_range=True)
            if data_t1_sub1 is not None:
                data_t1_sub1 = data_t1_sub1.transpose(2,0,1)
                data_t1 = np.concatenate((data_t1,
                                          resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192],
                                                 preserve_range=True)), axis=0) if data_t1.size else resize(
                    data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192], preserve_range=True)
            labels_sub1 = labels_sub1.transpose(2,0,1)
            labels = np.concatenate((labels, (resize(labels_sub1,[labels_sub1.shape[0],labels_sub1.shape[1],192],preserve_range=True)>0.5).astype(float)),axis=0) if labels.size else (resize(labels_sub1,[labels_sub1.shape[0],labels_sub1.shape[1],192],preserve_range=True)>0.5).astype(float)
            if is_weighted:
                GM_distance_sub1 = GM_distance_sub1.transpose(2, 0, 1)
                ventdistmap_sub1 = ventdistmap_sub1.transpose(2, 0, 1)
                GM_distance = np.concatenate((GM_distance, resize(GM_distance_sub1,[GM_distance_sub1.shape[0],GM_distance_sub1.shape[1],192],preserve_range=True)),axis=0) if GM_distance.size else resize(GM_distance_sub1,[GM_distance_sub1.shape[0],GM_distance_sub1.shape[1],192],preserve_range=True)
                ventdistmap = np.concatenate((ventdistmap, resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],ventdistmap_sub1.shape[1],192],preserve_range=True)),axis=0) if ventdistmap.size else resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],ventdistmap_sub1.shape[1],192],preserve_range=True)
        elif plane == 'sagittal':
            if data_sub1 is not None:
                data = np.concatenate((data, resize(data_sub1,[data_sub1.shape[0],192,120],preserve_range=True)),axis=0) if data.size else resize(data_sub1,[data_sub1.shape[0],192,120],preserve_range=True)
            if data_t1_sub1 is not None:
                data_t1 = np.concatenate((data_t1, resize(data_t1_sub1,[data_t1_sub1.shape[0],192,120],preserve_range=True)),axis=0) if data_t1.size else resize(data_t1_sub1,[data_t1_sub1.shape[0],192,120],preserve_range=True)
            labels = np.concatenate((labels, (resize(labels_sub1,[labels_sub1.shape[0],192,120],preserve_range=True)>0.5).astype(float)),axis=0) if labels.size else (resize(labels_sub1,[labels_sub1.shape[0],192,120],preserve_range=True)>0.5).astype(float)
            if is_weighted:
                GM_distance = np.concatenate((GM_distance, resize(GM_distance_sub1,[GM_distance_sub1.shape[0],192,120],preserve_range=True)),axis=0) if GM_distance.size else resize(GM_distance_sub1,[GM_distance_sub1.shape[0],192,120],preserve_range=True)
                ventdistmap = np.concatenate((ventdistmap, resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],192,120],preserve_range=True)),axis=0) if ventdistmap.size else resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],192,120],preserve_range=True)
        elif plane == 'coronal':
            if data_sub1 is not None:
                data_sub1 = data_sub1.transpose(1,0,2)
                data = np.concatenate(
                    (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80], preserve_range=True)),
                    axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80],
                                                     preserve_range=True)
            if data_t1_sub1 is not None:
                data_t1_sub1 = data_t1_sub1.transpose(1,0,2)
                data_t1 = np.concatenate((data_t1,
                                          resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80],
                                                 preserve_range=True)), axis=0) if data_t1.size else resize(
                    data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80], preserve_range=True)

            labels_sub1 = labels_sub1.transpose(1,0,2)
            labels = np.concatenate((labels, (resize(labels_sub1,[labels_sub1.shape[0],labels_sub1.shape[1],80],preserve_range=True)>0.5).astype(float)),axis=0) if labels.size else (resize(labels_sub1,[labels_sub1.shape[0],labels_sub1.shape[1],80],preserve_range=True)>0.5).astype(float)
            if is_weighted:
                GM_distance_sub1 = GM_distance_sub1.transpose(1, 0, 2)
                ventdistmap_sub1 = ventdistmap_sub1.transpose(1, 0, 2)
                GM_distance = np.concatenate((GM_distance, resize(GM_distance_sub1,[GM_distance_sub1.shape[0],GM_distance_sub1.shape[1],80],preserve_range=True)),axis=0) if GM_distance.size else resize(GM_distance_sub1,[GM_distance_sub1.shape[0],GM_distance_sub1.shape[1],80],preserve_range=True)
                ventdistmap = np.concatenate((ventdistmap, resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],ventdistmap_sub1.shape[1],80],preserve_range=True)),axis=0) if ventdistmap.size else resize(ventdistmap_sub1,[ventdistmap_sub1.shape[0],ventdistmap_sub1.shape[1],80],preserve_range=True)

    if data_sub1 is None:
        data = None
    else:
        data = np.tile(data,(1,1,1,1))
        data = data.transpose(1,2,3,0)
    if data_t1_sub1 is None:
        data_t1 = None
    else:
        data_t1 = np.tile(data_t1,(1,1,1,1))
        data_t1 = data_t1.transpose(1,2,3,0)
    labels = np.tile(labels,(1,1,1,1))
    labels = labels.transpose(1,2,3,0)

    input_data = {'flair': data, 't1': data_t1, 'label': labels, 'gmdist':None, 'ventdist':None}
    if is_weighted:
        input_data['gmdist'] = GM_distance
        input_data['ventdist'] = ventdistmap
    return input_data

def load_and_crop_data(data_path, weighted=True):
    '''
    Loads and crops the input data and distance maps (if required)
    :param data_path: dictionary of filepaths
    :param weighted: bool, whether to apply spatial weights in loss function
    :return: dictionary containing cropped arrays
    '''

    flair_path = data_path['flair_path']
    t1_path = data_path['t1_path']
    lab_path = data_path['gt_path']
    labels_sub = nib.load(lab_path).get_fdata()
    labels_sub[labels_sub > 1.5] = 0
    loaded_array = {'data_cropped': None,
                    'data_t1_cropped': None,
                    'label_cropped': None,
                    'gmdist_cropped': None,
                    'ventdist_cropped': None}

    if flair_path is None:
        labels_sub = nib.load(lab_path).get_fdata()
        labels_sub[labels_sub > 1.5] = 0
        data_t1_sub_org = nib.load(t1_path).get_fdata()
        _, coords = truenet_data_preprocessing.tight_crop_data(data_t1_sub_org)
        row_cent = coords[1] // 2 + coords[0]
        rowstart = np.amax([row_cent - 64, 0])
        rowend = np.amin([row_cent + 64, data_t1_sub_org.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_t1_sub1 = np.zeros([128, coords[3], coords[5]])
        data_t1_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
            data_t1_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_t1_sub1[:data_t1_sub_piece.shape[0], :data_t1_sub_piece.shape[1],
        :data_t1_sub_piece.shape[2]] = data_t1_sub_piece
        labels_sub1 = np.zeros([128, coords[3], coords[5]])
        labels_sub_piece = labels_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
        labels_sub1[:labels_sub_piece.shape[0], :labels_sub_piece.shape[1],
        :labels_sub_piece.shape[2]] = labels_sub_piece
        loaded_array['data_t1_cropped'] = data_t1_sub1
        loaded_array['label_cropped'] = labels_sub1
    else:
        labels_sub = nib.load(lab_path).get_fdata()
        labels_sub[labels_sub > 1.5] = 0
        data_sub_org = nib.load(flair_path).get_fdata()
        _, coords = truenet_data_preprocessing.tight_crop_data(data_sub_org)
        row_cent = coords[1] // 2 + coords[0]
        rowstart = np.amax([row_cent - 64, 0])
        rowend = np.amin([row_cent + 64, data_sub_org.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_sub1 = np.zeros([128, coords[3], coords[5]])
        data_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
            data_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_sub1[:data_sub_piece.shape[0], :data_sub_piece.shape[1], :data_sub_piece.shape[2]] = data_sub_piece
        labels_sub1 = np.zeros([128, coords[3], coords[5]])
        labels_sub_piece = labels_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
        labels_sub1[:labels_sub_piece.shape[0], :labels_sub_piece.shape[1],
        :labels_sub_piece.shape[2]] = labels_sub_piece
        loaded_array['data_cropped'] = data_sub1
        loaded_array['label_cropped'] = labels_sub1
        if t1_path is not None:
            data_t1_sub_org = nib.load(t1_path).get_fdata()
            data_t1_sub1 = np.zeros([128, coords[3], coords[5]])
            data_t1_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
                data_t1_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
            data_t1_sub1[:data_t1_sub_piece.shape[0], :data_t1_sub_piece.shape[1],
            :data_t1_sub_piece.shape[2]] = data_t1_sub_piece
            loaded_array['data_t1_cropped'] = data_t1_sub1
    if weighted:
        gmdist_path = data_path['gmdist_path']
        ventdist_path = data_path['ventdist_path']
        GM_distance_sub = nib.load(gmdist_path).get_fdata()
        ventdistmap_sub = nib.load(ventdist_path).get_fdata()
        GM_distance_sub1 = np.zeros([128,coords[3],coords[5]])
        ventdistmap_sub1 = np.zeros([128,coords[3],coords[5]])
        GM_distance_sub_piece = GM_distance_sub[rowstart:rowend,colstart:colend,stackstart:stackend]
        ventdistmap_sub_piece = ventdistmap_sub[rowstart:rowend,colstart:colend,stackstart:stackend]
        GM_distance_sub1[:GM_distance_sub_piece.shape[0],:GM_distance_sub_piece.shape[1],:GM_distance_sub_piece.shape[2]] = GM_distance_sub_piece
        ventdistmap_sub1[:ventdistmap_sub_piece.shape[0],:ventdistmap_sub_piece.shape[1],:ventdistmap_sub_piece.shape[2]] = ventdistmap_sub_piece
        loaded_array['gmdist_cropped'] = GM_distance_sub1
        loaded_array['ventdist_cropped'] = ventdistmap_sub1

    return loaded_array


def get_slices_from_data_with_aug(loaded_data_array, af=2, plane='axial', test=0, weighted=True):
    '''
    getting the final stack of slices after data augmentation (if chosen) to form datasets.
    :param loaded_data_array: dictionary of reshaped input arrays
    :param af: int, augmentation factor
    :param plane: str, acquisition plane
    :param test: binary, if test == 1, no data sugmentation will be applied
    :param weighted: bool, whether to use spatial weights in the loss function
    :return:
    '''
    data = loaded_data_array['flair']
    data_t1 = loaded_data_array['t1']
    labels = loaded_data_array['label']
    labels = (labels[:, :, :, 0]==1).astype(float)

    if plane == 'sagittal':
        aug_factor = af
    elif plane == 'coronal':
        aug_factor = af
    elif plane == 'axial':
        aug_factor = af+1

    if weighted:
        GM_distance = loaded_data_array['gmdist']
        ventdistmap = loaded_data_array['ventdist']
        gm_distance = 10*GM_distance**0.33
        ventdistmap = 6*(ventdistmap**.5)

        if test == 0:
            data, data_t1, labels, GM_distance, ventdistmap = perform_augmentation_withdistmaps(data, data_t1,
                                                                                                    labels, gm_distance,
                                                                                                    ventdistmap,
                                                                                                    af=aug_factor)
        if data is not None and data_t1 is not None:
            tr_data = np.concatenate((data, data_t1), axis=-1)
        elif data is not None and data_t1 is None:
            tr_data = data
        else:
            tr_data = data_t1
        data2d = [tr_data, labels, GM_distance, ventdistmap]
    else:
        if test == 0:
            data, data_t1, labels = perform_augmentation(data, data_t1, labels, af=aug_factor)
        if data is not None and data_t1 is not None:
            tr_data = np.concatenate((data, data_t1), axis=-1)
        elif data is not None and data_t1 is None:
            tr_data = data
        else:
            tr_data = data_t1
        data2d = [tr_data, labels]
    return data2d


def perform_augmentation_withdistmaps(otr, otr_t1, otr_labs, otr_gmdist, otr_ventdistmap, af=2):
    '''
    Perform augmentation including distance maps
    :param otr: FLAIR 4D [N, H, W, 1]
    :param otr_t1: T1 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param otr_gmdist: GM distance 3D [N, H, W]
    :param otr_ventdistmap: ventricle distance 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img_t1_list = []
    augmented_mseg_list = []
    augmented_gmdist_list = []
    augmented_ventdist_list = []
    if otr is not None:
        inpp = otr
    else:
        inpp = otr_t1
    for i in range(0, af):
        for id in range(inpp.shape[0]):
            if otr is not None:
                image = otr[id, :, :, 0]
            else:
                image = np.zeros_like(inpp[id, :, :, 0])
            if otr_t1 is not None:
                image_t1 = otr_t1[id, :, :, 0]
            else:
                image_t1 = np.zeros_like(inpp[id, :, :, 0])
            manmask = otr_labs[id,:,:]
            gmdist = otr_gmdist[id,:,:]
            ventdist = otr_ventdistmap[id,:,:]
            augmented_img, augmented_img_t1, augmented_manseg, augmented_gmdist, augmented_ventdist = truenet_augmentation.augment_distmaps(image,image_t1,manmask,gmdist,ventdist)
            augmented_img_list.append(augmented_img)
            augmented_img_t1_list.append(augmented_img_t1)
            augmented_mseg_list.append(augmented_manseg)
            augmented_gmdist_list.append(augmented_gmdist)
            augmented_ventdist_list.append(augmented_ventdist)
    augmented_img = np.array(augmented_img_list)
    augmented_img_t1 = np.array(augmented_img_t1_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_gmdist = np.array(augmented_gmdist_list)
    augmented_ventdist = np.array(augmented_ventdist_list)
    augmented_img = np.reshape(augmented_img,[-1,inpp.shape[1],inpp.shape[2]])
    augmented_img_t1 = np.reshape(augmented_img_t1,[-1,inpp.shape[1],inpp.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg,[-1,inpp.shape[1],inpp.shape[2]])
    augmented_gmdist = np.reshape(augmented_gmdist,[-1,inpp.shape[1],inpp.shape[2]])
    augmented_ventdist = np.reshape(augmented_ventdist,[-1,inpp.shape[1],inpp.shape[2]])
    augmented_img = np.tile(augmented_img,(1,1,1,1))
    augmented_imgs = augmented_img.transpose(1,2,3,0)
    augmented_img_t1 = np.tile(augmented_img_t1,(1,1,1,1))
    augmented_imgs_t1 = augmented_img_t1.transpose(1,2,3,0)
    if otr is not None:
        otr_aug = np.concatenate((otr,augmented_imgs),axis=0)
    else:
        otr_aug = None
    if otr_t1 is not None:
        otr_aug_t1 = np.concatenate((otr_t1,augmented_imgs_t1),axis=0)
    else:
        otr_aug_t1 = None
    otr_labs = np.concatenate((otr_labs,augmented_mseg),axis = 0)
    otr_gmdist = np.concatenate((otr_gmdist,augmented_gmdist),axis = 0)
    otr_ventdistmap = np.concatenate((otr_ventdistmap,augmented_ventdist),axis = 0)
    return otr_aug, otr_aug_t1, otr_labs, otr_gmdist, otr_ventdistmap


def perform_augmentation(otr,otr_t1,otr_labs,af = 2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: FLAIR 4D [N, H, W, 1]
    :param otr_t1: T1 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img_t1_list = []
    augmented_mseg_list = []
    if otr is not None:
        inpp = otr
    else:
        inpp = otr_t1
    for i in range(0, af):
        for id in range(inpp.shape[0]):
            if otr is not None:
                image = otr[id, :, :, 0]
            else:
                image = np.zeros_like(inpp[id, :, :, 0])
            if otr_t1 is not None:
                image_t1 = otr_t1[id, :, :, 0]
            else:
                image_t1 = np.zeros_like(inpp[id, :, :, 0])
            manmask = otr_labs[id,:,:]
            augmented_img, augmented_img_t1, augmented_manseg = truenet_augmentation.augment(image,image_t1,manmask)
            augmented_img_list.append(augmented_img)
            augmented_img_t1_list.append(augmented_img_t1)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_img_t1 = np.array(augmented_img_t1_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img,[-1,otr.shape[1],otr.shape[2]])
    augmented_img_t1 = np.reshape(augmented_img_t1,[-1,otr.shape[1],otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg,[-1,otr.shape[1],otr.shape[2]])
    augmented_img = np.tile(augmented_img,(1,1,1,1))
    augmented_imgs = augmented_img.transpose(1,2,3,0)
    augmented_img_t1 = np.tile(augmented_img_t1,(1,1,1,1))
    augmented_imgs_t1 = augmented_img_t1.transpose(1,2,3,0)
    if otr is not None:
        otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    else:
        otr_aug = None
    if otr_t1 is not None:
        otr_aug_t1 = np.concatenate((otr_t1, augmented_imgs_t1), axis=0)
    else:
        otr_aug_t1 = None
    otr_labs = np.concatenate((otr_labs,augmented_mseg),axis=0)
    return otr_aug, otr_aug_t1, otr_labs


def create_test_data_array(names, plane='axial'):
    '''
    Create the input stack of 2D slices reshaped to required dimensions
    :param names: list of dictionaries containing filepaths
    :param plane: acquisition plane
    :return: dictionary of input arrays
    '''
    data = np.array([])
    data_t1 = np.array([])

    for i in range(len(names)):
        array_loaded = load_and_crop_test_data(names[i])
        data_sub1 = array_loaded['data_cropped']
        data_t1_sub1 = array_loaded['data_t1_cropped']
        if plane == 'axial':
            if data_sub1 is not None:
                data_sub1 = data_sub1.transpose(2, 0, 1)
                data = np.concatenate(
                    (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192], preserve_range=True)),
                    axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192],
                                                     preserve_range=True)
            else:
                data = None
            if data_t1_sub1 is not None:
                data_t1_sub1 = data_t1_sub1.transpose(2, 0, 1)
                data_t1 = np.concatenate((data_t1,
                                          resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192],
                                                 preserve_range=True)), axis=0) if data_t1.size else resize(
                    data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192], preserve_range=True)
            else:
                data_t1 = None
        elif plane == 'sagittal':
            if data_sub1 is not None:
                data = np.concatenate((data, resize(data_sub1, [data_sub1.shape[0], 192, 120], preserve_range=True)),
                                      axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], 192, 120],
                                                                       preserve_range=True)
            else:
                data = None
            if data_t1_sub1 is not None:
                data_t1 = np.concatenate(
                    (data_t1, resize(data_t1_sub1, [data_t1_sub1.shape[0], 192, 120], preserve_range=True)),
                    axis=0) if data_t1.size else resize(data_t1_sub1, [data_t1_sub1.shape[0], 192, 120],
                                                        preserve_range=True)
            else:
                data_t1 = None
        elif plane == 'coronal':
            if data_sub1 is not None:
                data_sub1 = data_sub1.transpose(1, 0, 2)
                data = np.concatenate(
                    (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80], preserve_range=True)),
                    axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80],
                                                     preserve_range=True)
            else:
                data = None
            if data_t1_sub1 is not None:
                data_t1_sub1 = data_t1_sub1.transpose(1, 0, 2)
                data_t1 = np.concatenate((data_t1,
                                          resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80],
                                                 preserve_range=True)), axis=0) if data_t1.size else resize(
                    data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80], preserve_range=True)
            else:
                data_t1 = None

    if data is not None and data_t1 is not None:
        data = np.tile(data, (1, 1, 1, 1))
        data = data.transpose(1, 2, 3, 0)
        data_t1 = np.tile(data_t1, (1, 1, 1, 1))
        data_t1 = data_t1.transpose(1, 2, 3, 0)
        tr_data = np.concatenate((data, data_t1), axis=-1)
    elif data is not None and data_t1 is None:
        data = np.tile(data, (1, 1, 1, 1))
        data = data.transpose(1, 2, 3, 0)
        tr_data = data
    else:
        data_t1 = np.tile(data_t1, (1, 1, 1, 1))
        data_t1 = data_t1.transpose(1, 2, 3, 0)
        tr_data = data_t1
    data2d = [tr_data]
    return data2d


def load_and_crop_test_data(data_path):
    '''
    Loads and crops the input data
    :param data_path: dictionary of filepaths
    :return: dictionary containing cropped arrays
    '''
    flair_path = data_path['flair_path']
    t1_path = data_path['t1_path']
    loaded_array = {'data_cropped': None,
                    'data_t1_cropped': None,
                    'gmdist_cropped': None,
                    'ventdist_cropped': None}

    if flair_path is None:
        data_t1_sub_org = nib.load(t1_path).get_fdata()
        _, coords = truenet_data_preprocessing.tight_crop_data(data_t1_sub_org)
        row_cent = coords[1] // 2 + coords[0]
        rowstart = np.amax([row_cent - 64, 0])
        rowend = np.amin([row_cent + 64, data_t1_sub_org.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_t1_sub1 = np.zeros([128, coords[3], coords[5]])
        data_t1_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
            data_t1_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_t1_sub1[:data_t1_sub_piece.shape[0], :data_t1_sub_piece.shape[1],
        :data_t1_sub_piece.shape[2]] = data_t1_sub_piece
        loaded_array['data_t1_cropped'] = data_t1_sub1
    else:
        data_sub_org = nib.load(flair_path).get_fdata()
        _, coords = truenet_data_preprocessing.tight_crop_data(data_sub_org)
        row_cent = coords[1] // 2 + coords[0]
        rowstart = np.amax([row_cent - 64, 0])
        rowend = np.amin([row_cent + 64, data_sub_org.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_sub1 = np.zeros([128, coords[3], coords[5]])
        data_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
            data_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_sub1[:data_sub_piece.shape[0], :data_sub_piece.shape[1], :data_sub_piece.shape[2]] = data_sub_piece
        loaded_array['data_cropped'] = data_sub1
        if t1_path is not None:
            data_t1_sub_org = nib.load(t1_path).get_fdata()
            data_t1_sub1 = np.zeros([128, coords[3], coords[5]])
            data_t1_sub_piece = truenet_data_preprocessing.preprocess_data_gauss(
                data_t1_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
            data_t1_sub1[:data_t1_sub_piece.shape[0], :data_t1_sub_piece.shape[1],
            :data_t1_sub_piece.shape[2]] = data_t1_sub_piece
            loaded_array['data_t1_cropped'] = data_t1_sub1
    return loaded_array
