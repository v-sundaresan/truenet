import functools as ft

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

    models = {}

    print('Total number of model parameters to train', flush=True)
    for plane in ['axial', 'sagittal', 'coronal']:
        model = truenet_model.TrUENet(n_channels=num_channels, n_classes=nclass, init_channels=64, plane=plane)
        model.to(device=device)
        model = nn.DataParallel(model)
        models[plane] = model
        print(f'{plane} model: ', str(sum([p.numel() for p in model.parameters()])), flush=True)

    if optim_type == 'adam':
        optimizer = ft.partial(optim.Adam, lr=lrt, eps=tr_params['Epsilon'])
    elif optim_type == 'sgd':
        optimizer = ft.partial(optim.SGD, lr=lrt, momentum=tr_params['Momentum'])
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    optimizers = {}
    for plane, model in models.items():
        optimizers[plane] = optimizer(
            filter(lambda p: p.requires_grad, model.parameters()))

    if nclass == 2:
        criterion = truenet_loss_functions.CombinedLoss()
    else:
        criterion = truenet_loss_functions.CombinedMultiLoss(nclasses=nclass)

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)),1)
    train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(
        sub_name_dicts, num_val_subs)

    if type(milestones) != list:
        milestones = [milestones]

    if req_plane == 'all': planes = ['axial', 'sagittal', 'coronal']
    else:                  planes = [req_plane]

    for plane in planes:
        model     = models[plane]
        optimizer = optimizers[plane]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=gamma, last_epoch=-1)
        truenet_train.train_truenet(
            train_name_dicts, val_name_dicts, model, criterion, optimizer, scheduler,
            tr_params, device, mode=plane, augment=aug, weighted=weighted,
            save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case,
            verbose=verbose, dir_checkpoint=dir_cp)
