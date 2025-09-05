import torch
import torch.nn as nn
from torch import optim
import functools as ft
import os.path as op
from truenet.true_net import (truenet_loss_functions,
                              truenet_model, truenet_train)
from truenet.utils import (truenet_utils)

#=========================================================================================
# Truenet fine_tuning function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


def main(sub_name_dicts, device, ft_params, aug=True, weighted=True, save_cp=True, save_wei=True, save_case='best',
         verbose=True, model_dir=None, dir_cp=None):
    '''
    The main function for fine-tuning the model
    :param sub_name_dicts: list of dictionaries containing subject filepaths for fine-tuning
    :param device: Pytorch device
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

    # number of channels (T1/FLAIR) present in input
    input_channels = ft_params['Numchannels']
    model_name     = ft_params['Modelname']

    # peek at one of the model files to identify
    # expected number of input channels and output
    # classes
    model_path          = op.join(model_dir, model_name + '_axial.pth')
    nclasses, nchannels = truenet_utils.peek_model(model_path)

    if nchannels != input_channels:
        raise ImportError(f'Model {model_name} was trained on {nchannels} channels '
                          f'(T1/FLAIR), but input data contains {input_channels} channels!')

    models = {}

    for plane in ['axial', 'sagittal', 'coronal']:
        model_path = f'{model_dir}/{model_name}_{plane}.pth'
        model = truenet_model.TrUENet(n_channels=nchannels, n_classes=nclasses, init_channels=64, plane=plane)
        model.to(device=device)
        model = nn.DataParallel(model)
        model = truenet_utils.load_model(model_path, model, device)
        models[plane] = model

    layers_to_ft = ft_params['Finetuning_layers']  # list of numbers [1,8]
    optim_type = ft_params['Optimizer']  # adam, sgd
    milestones = ft_params['LR_Milestones']  # list of integers [1, N]
    gamma = ft_params['LR_red_factor']  # scalar (0,1)
    ft_lrt = ft_params['Finetuning_learning_rate']  # scalar (0,1)
    req_plane = ft_params['Acq_plane']  # string ('axial', 'sagittal', 'coronal', 'all')
    train_prop = ft_params['Train_prop']  # scale (0,1)

    if type(milestones) != list:
        milestones = [milestones]

    if type(layers_to_ft) != list:
        layers_to_ft = [layers_to_ft]

    print('Total number of model parameters', flush=True)
    print('Axial model: ', str(sum([p.numel() for p in models['axial'].parameters()])), flush=True)
    print('Sagittal model: ', str(sum([p.numel() for p in models['sagittal'].parameters()])), flush=True)
    print('Coronal model: ', str(sum([p.numel() for p in models['coronal'].parameters()])), flush=True)

    print('Total number of trainable parameters', flush=True)
    for plane, model in list(models.items()):
        model = truenet_utils.freeze_layer_for_finetuning(model, layers_to_ft, verbose=verbose)
        models[plane] = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([p.numel() for p in model_parameters])
        print(f'{plane} model: ', str(params), flush=True)

    if optim_type == 'adam':
        optimizer = ft.partial(optim.Adam, lr=ft_lrt, eps=ft_params['Epsilon'])
    elif optim_type == 'sgd':
        optimizer = ft.partial(optim.Adam, lr=ft_lrt, momentum=ft_params['Momentum'])
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    optimizers = {}

    for plane, model in models.items():
        optimizers[plane] = optimizer(
            filter(lambda p: p.requires_grad, model.parameters()))

    if nclasses == 2:
        criterion = truenet_loss_functions.CombinedLoss()
    else:
        criterion = truenet_loss_functions.CombinedMultiLoss(nclasses=nclasses)

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)), 1)
    train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(
        sub_name_dicts, num_val_subs)

    if req_plane == 'all': planes = ['axial', 'sagittal', 'coronal']
    else:                  planes = [req_plane]

    for plane in planes:
        optimizer = optimizers[plane]
        model     = models[plane]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=gamma, last_epoch=-1)

        truenet_train.train_truenet(
            train_name_dicts, val_name_dicts, model, criterion,
            optimizer, scheduler, ft_params, device, mode=plane,
            augment=aug, weighted=weighted, save_checkpoint=save_cp,
            save_weights=save_wei, save_case=save_case, verbose=verbose,
            dir_checkpoint=dir_cp)

    print('Model Fine-tuning done!', flush=True)
