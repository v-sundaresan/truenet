from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
import os
from truenet.true_net import (truenet_loss_functions,
                              truenet_model, truenet_train)
from truenet.utils import (truenet_utils)

#=========================================================================================
# Truenet fine_tuning function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================

def main(sub_name_dicts, ft_params, aug=True, weighted=True, save_cp=True, save_wei=True, save_case='best',
         verbose=True, model_dir=None, dir_cp=None):
    '''
    The main function for fine-tuning the model
    :param sub_name_dicts: list of dictionaries containing subject filepaths for fine-tuning
    :param ft_params: dictionary of fine-tuning parameters
    :param aug: bool, whether to do data augmentation
    :param weighted: bool, whether to use spatial weights in loss function
    :param save_cp: bool, whether to save checkpoint
    :param save_wei: bool, whether to save weights alone or the full model
    :param save_case: str, condition for saving the CP
    :param verbose: bool, display debug messages
    :param model_dir: str, filepath containing pretrained model
    :param dir_cp: str, filepath for saving the model
    '''
    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for fine-tuning cannot be less than 5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclass = ft_params['Nclass']
    model_axial = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='axial')
    model_sagittal = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='sagittal')
    model_coronal = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='coronal')

    model_axial.to(device=device)
    model_sagittal.to(device=device)
    model_coronal.to(device=device)
    model_axial = nn.DataParallel(model_axial)
    model_sagittal = nn.DataParallel(model_sagittal)
    model_coronal = nn.DataParallel(model_coronal)

    load_case = ft_params['Load_type']
    pretrained = ft_params['Pretrained']

    if pretrained:
        if load_case == 'last':
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_beforeES_axial.pth'))
            model_axial.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_beforeES_sagittal.pth'))
            model_sagittal.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_beforeES_coronal.pth'))
            model_coronal.load_state_dict(cp['model_state_dict'])
        elif load_case == 'best':
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_bestdice_axial.pth'))
            model_axial.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_bestdice_sagittal.pth'))
            model_sagittal.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, 'Truenet_model_bestdice_coronal.pth'))
            model_coronal.load_state_dict(cp['model_state_dict'])
        elif load_case == 'specific':
            cpn = ft_params['EveryN']
            try:
                cp = torch.load(os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_axial.pth'))
                model_axial.load_state_dict(cp['model_state_dict'])
                cp = torch.load(os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_sagittal.pth'))
                model_sagittal.load_state_dict(cp['model_state_dict'])
                cp = torch.load(os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_coronal.pth'))
                model_coronal.load_state_dict(cp['model_state_dict'])
            except ImportError:
                raise ImportError(
                    'Incorrect N value provided for the available pretrained models')
        else:
            raise ValueError("Invalid loading condition provided! Valid options: best, specific, last")
    else:
        model_name = ft_params['Modelname']
        try:
            cp = torch.load(os.path.join(model_dir, model_name + '_axial.pth'))
            model_axial.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, model_name + '_sagittal.pth'))
            model_sagittal.load_state_dict(cp['model_state_dict'])
            cp = torch.load(os.path.join(model_dir, model_name + '_coronal.pth'))
            model_coronal.load_state_dict(cp['model_state_dict'])
        except ImportError:
            raise ImportError('In directory ' + model_dir + ', ' + model_name + '_axial.pth/' +
                              model_name + '_sagittal.pth/' + model_name + '_coronal.pth/ ' +
                              'does not appear to be a valid model file')

    layers_to_ft = ft_params['Finetuning_layers'] # list of numbers [1,8]
    optim_type = ft_params['Optimizer'] # adam, sgd
    milestones = ft_params['LR_Milestones']# list of integers [1, N]
    gamma = ft_params['LR_red_factor'] # scalar (0,1)
    ft_lrt = ft_params['Finetuning_learning_rate'] # scalar (0,1)
    req_plane = ft_params['Acq_plane'] # string ('axial', 'sagittal', 'coronal', 'all')
    train_prop = ft_params['Train_prop'] # scale (0,1)

    if type(milestones) != list:
        milestones = [milestones]

    if type(layers_to_ft) != list:
        layers_to_ft = [layers_to_ft]

    print('Total number of model parameters', flush=True)
    print('Axial model: ', str(sum([p.numel() for p in model_axial.parameters()])), flush=True)
    print('Sagittal model: ', str(sum([p.numel() for p in model_sagittal.parameters()])), flush=True)
    print('Coronal model: ', str(sum([p.numel() for p in model_coronal.parameters()])), flush=True)
    
    model_axial = truenet_utils.freeze_layer_for_finetuning(model_axial, layers_to_ft, verbose=verbose)
    model_sagittal = truenet_utils.freeze_layer_for_finetuning(model_sagittal, layers_to_ft, verbose=verbose)
    model_coronal = truenet_utils.freeze_layer_for_finetuning(model_coronal, layers_to_ft, verbose=verbose)
    model_axial.to(device=device)
    model_sagittal.to(device=device)
    model_coronal.to(device=device)
    
    print('Total number of trainable parameters', flush=True)
    model_parameters = filter(lambda p: p.requires_grad, model_axial.parameters())
    params = sum([p.numel() for p in model_parameters])
    print('Axial model: ', str(params), flush=True)
    model_parameters = filter(lambda p: p.requires_grad, model_sagittal.parameters())
    params = sum([p.numel() for p in model_parameters])
    print('Sagittal model: ', str(params), flush=True)
    model_parameters = filter(lambda p: p.requires_grad, model_coronal.parameters())
    params = sum([p.numel() for p in model_parameters])
    print('Coronal model: ', str(params), flush=True)
    
    if optim_type == 'adam':
        epsilon = ft_params['Epsilon']
        optimizer_axial = optim.Adam(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=ft_lrt, eps=epsilon)
        optimizer_sagittal = optim.Adam(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=ft_lrt, eps=epsilon)
        optimizer_coronal = optim.Adam(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=ft_lrt, eps=epsilon)
    elif optim_type == 'sgd':
        moment = ft_params['Momentum']
        optimizer_axial = optim.SGD(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=ft_lrt, momentum=moment)
        optimizer_sagittal = optim.SGD(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=ft_lrt, momentum=moment)
        optimizer_coronal = optim.SGD(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=ft_lrt, momentum=moment)
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        
    if nclass == 2:
        criterion = truenet_loss_functions.CombinedLoss()
    else:
        criterion = truenet_loss_functions.CombinedMultiLoss(nclasses=nclass)
    
    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)), 1)
    train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(sub_name_dicts,
                                                                                           num_val_subs)

    if req_plane == 'all' or req_plane == 'axial':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_axial, milestones, gamma=gamma, last_epoch=-1)
        model_axial = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_axial, criterion, optimizer_axial, scheduler, 
                                ft_params, device, mode='axial', augment=aug, weighted=weighted, 
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)
        
    if req_plane == 'all' or req_plane == 'sagittal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sagittal, milestones, gamma=gamma, last_epoch=-1)
        model_sagittal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_sagittal, criterion, optimizer_sagittal, scheduler, 
                                ft_params, device, mode='sagittal', augment=aug, weighted=weighted, 
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)
        
    if req_plane == 'all' or req_plane == 'coronal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_coronal, milestones, gamma=gamma, last_epoch=-1)
        model_coronal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_coronal, criterion, optimizer_coronal, scheduler, 
                                ft_params, device, mode='coronal', augment=aug, weighted=weighted, 
                                save_checkpoint=save_cp, save_weights=save_wei, save_case=save_case, verbose=verbose, 
                                dir_checkpoint=dir_cp)

    print('Model Fine-tuning done!', flush=True)



