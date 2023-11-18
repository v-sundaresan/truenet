from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from truenet.true_net import (truenet_loss_functions,
                              truenet_model, truenet_train)
from truenet.utils import truenet_utils

#=========================================================================================
# Truenet main training function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def main(sub_name_dicts, tr_params, aug=True, weighted=True,
         save_cp=True, save_wei=True, save_case='last', verbose=True, dir_cp=None):
    '''
    The main training function
    :param sub_name_dicts: list of dictionaries containing training filpaths
    :param tr_params: dictionary of training parameters
    :param aug: bool, perform data augmentation
    :param weighted: bool, apply spatial weights in the loss function
    :param save_cp: bool, save checkpoints
    :param save_wei: bool, if False, the whole model will be saved
    :param save_case: str, condition for saving the checkpoint
    :param verbose: bool, display debug messages
    :param dir_cp: str, directory for saving model/weights
    :return: trained model
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for training cannot be less than 5"
    
    optim_type = tr_params['Optimizer'] # adam, sgd
    milestones = tr_params['LR_Milestones']# list of integers [1, N]
    gamma = tr_params['LR_red_factor'] # scalar (0,1)
    lrt = tr_params['Learning_rate'] # scalar (0,1)
    req_plane = tr_params['Acq_plane'] # string ('axial', 'sagittal', 'coronal', 'all')
    train_prop = tr_params['Train_prop'] # scale (0,1)
    nclass = tr_params['Nclass']
    num_channels = tr_params['Numchannels']

    model_axial = truenet_model.TrUENet(n_channels=num_channels, n_classes=nclass, init_channels=64, plane='axial')
    model_sagittal = truenet_model.TrUENet(n_channels=num_channels, n_classes=nclass, init_channels=64, plane='sagittal')
    model_coronal = truenet_model.TrUENet(n_channels=num_channels, n_classes=nclass, init_channels=64, plane='coronal')

    model_axial.to(device=device)
    model_sagittal.to(device=device)
    model_coronal.to(device=device)
    model_axial = nn.DataParallel(model_axial)
    model_sagittal = nn.DataParallel(model_sagittal)
    model_coronal = nn.DataParallel(model_coronal)
    
    print('Total number of model parameters to train', flush=True)
    print('Axial model: ', str(sum([p.numel() for p in model_axial.parameters()])), flush=True)
    print('Sagittal model: ', str(sum([p.numel() for p in model_sagittal.parameters()])), flush=True)
    print('Coronal model: ', str(sum([p.numel() for p in model_coronal.parameters()])), flush=True)
    
    if optim_type == 'adam':
        epsilon = tr_params['Epsilon']
        optimizer_axial = optim.Adam(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=lrt, eps=epsilon)
        optimizer_sagittal = optim.Adam(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=lrt, eps=epsilon)
        optimizer_coronal = optim.Adam(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=lrt, eps=epsilon)
    elif optim_type == 'sgd':
        moment = tr_params['Momentum']
        optimizer_axial = optim.SGD(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=lrt, momentum=moment)
        optimizer_sagittal = optim.SGD(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=lrt, momentum=moment)
        optimizer_coronal = optim.SGD(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=lrt, momentum=moment)
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        
    if nclass == 2:
        criterion = truenet_loss_functions.CombinedLoss()
    else:
        criterion = truenet_loss_functions.CombinedMultiLoss(nclasses=nclass)
    
    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)
        
    num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)),1)
    train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(sub_name_dicts,
                                                                                           num_val_subs)
    if type(milestones) != list:
        milestones = [milestones]

    if req_plane == 'all' or req_plane == 'axial':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_axial, milestones, gamma=gamma, last_epoch=-1)
        model_axial = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_axial, criterion, optimizer_axial, scheduler, 
                                tr_params, device, mode='axial', augment=aug, weighted=weighted,
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)
        
    if req_plane == 'all' or req_plane == 'sagittal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sagittal, milestones, gamma=gamma, last_epoch=-1)
        model_sagittal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_sagittal, criterion, optimizer_sagittal, scheduler, 
                                tr_params, device, mode='sagittal', augment=aug, weighted=weighted,
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)
        
    if req_plane == 'all' or req_plane == 'coronal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_coronal, milestones, gamma=gamma, last_epoch=-1)
        model_coronal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_coronal, criterion, optimizer_coronal, scheduler, 
                                tr_params, device, mode='coronal', augment=aug, weighted=weighted,
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)
    models = [model_axial, model_sagittal, model_coronal]
    return models


