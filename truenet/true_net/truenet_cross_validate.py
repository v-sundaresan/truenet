import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import functools as ft
import nibabel as nib
from truenet.true_net import (truenet_loss_functions,
                              truenet_model, truenet_train, truenet_evaluate,
                              truenet_data_postprocessing)
from truenet.utils import truenet_utils

#=========================================================================================
# Truenet cross-validation function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def main(sub_name_dicts, cv_params, aug=True, weighted=True, intermediate=False,
         save_cp=False, save_wei=True, save_case='best', verbose=True, dir_cp=None, output_dir=None):
    '''
    The main function for leave-one-out validation of Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param cv_params: dictionary of LOO paramaters
    :param aug: bool, whether to do data augmentation
    :param weighted: bool, whether to use spatial weights in loss function
    :param intermediate: bool, whether to save intermediate results
    :param save_cp: bool, whether to save checkpoint
    :param save_wei: bool, whether to save weights alone or the full model
    :param save_case: str, condition for saving the CP
    :param verbose: bool, display debug messages
    :param dir_cp: str, filepath for saving the model
    :param output_dir: str, filepath for saving the output predictions
    '''

    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for Leave-one-out validation cannot be less than 5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclass = cv_params['Nclass']
    num_channels = cv_params['Numchannels']

    models = {}
    for plane in ['axial', 'sagittal', 'coronal']:
        model = truenet_model.TrUENet(n_channels=num_channels, n_classes=nclass, init_channels=64, plane=plane)
        model.to(device=device)
        model = nn.DataParallel(model)
        models[plane] = model

    optim_type = cv_params['Optimizer']  # adam, sgd
    milestones = cv_params['LR_Milestones']  # list of integers [1, N]
    gamma = cv_params['LR_red_factor']  # scalar (0,1)
    lrt = cv_params['Learning_rate']  # scalar (0,1)
    train_prop = cv_params['Train_prop']  # scalar (0,1)
    fold = cv_params['fold']  # scalar [1, N]
    res_fold = cv_params['res_fold']  # scalar [1, N]

    res_fold = res_fold - 1

    test_subs_per_fold = max(int(np.round(len(sub_name_dicts) / fold)), 1)

    if type(milestones) != list:
        milestones = [milestones]

    if optim_type == 'adam':
        optimizer = ft.partial(optim.Adam, lr=lrt, eps=cv_params['Epsilon'])
    elif optim_type == 'sgd':
        optimizer = ft.partial(optim.SGD, lr=lrt, momentum=cv_params['Momentum'])
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    optimizers = {}
    for plane, model in models.items():
        optimizers[plane] = optimizer(model.parameters())

    if nclass == 2:
        criterion = truenet_loss_functions.CombinedLoss()
    else:
        criterion = truenet_loss_functions.CombinedMultiLoss(nclasses=nclass)

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    for fld in range(res_fold, fold):
        if verbose:
            print('Training models for fold ' + str(fld+1) + '...', flush=True)

        if fld == (fold - 1):
            test_ids = np.arange(fld * test_subs_per_fold, len(sub_name_dicts))
            test_sub_dicts = [sub_name_dicts[i] for i in test_ids]
        else:
            test_ids = np.arange(fld * test_subs_per_fold, (fld+1) * test_subs_per_fold)
            test_sub_dicts = [sub_name_dicts[i] for i in test_ids]

        rem_sub_ids = np.setdiff1d(np.arange(len(sub_name_dicts)),test_ids)
        rem_sub_name_dicts = [sub_name_dicts[idx] for idx in rem_sub_ids]
        num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)),1)
        train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(
            rem_sub_name_dicts, num_val_subs)
        if save_cp:
            dir_cp = os.path.join(dir_cp, 'fold' + str(fld+1) + '_models')
            os.mkdir(dir_cp)

        for plane, model in list(models.items()):
            optimizer = optimizers[plane]

            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones, gamma=gamma, last_epoch=-1)
            models[plane] = truenet_train.train_truenet(
                train_name_dicts, val_name_dicts, model, criterion,
                optimizer, scheduler, cv_params, device, mode=plane,
                augment=aug, weighted=weighted, save_checkpoint=save_cp,
                save_weights=save_wei, save_case=save_case, verbose=verbose,
                dir_checkpoint=dir_cp)

        if verbose:
            print('Predicting outputs for subjects in fold ' + str(fld+1) + '...', flush=True)

        for sub in range(len(test_sub_dicts)):
            if verbose:
                print('Predicting for subject ' + str(sub + 1) + '...', flush=True)
            test_sub_dict = [test_sub_dicts[sub]]
            basename = test_sub_dict[0]['basename']
            probs_combined = []
            flair_path = test_sub_dict[0]['flair_path']
            flair_hdr = nib.load(flair_path).header

            for plane, model in models.items():

                probs = truenet_evaluate.evaluate_truenet(
                    test_sub_dict, model, cv_params, device,
                    mode=plane, verbose=verbose)
                probs = truenet_data_postprocessing.resize_to_original_size(
                    probs, test_sub_dict, plane=plane)
                probs_combined.append(probs)

                if intermediate:
                    save_path = truenet_utils.addSuffix(f'{output_dir}/Predicted_probmap_truenet_{basename}_{plane}')
                    preds = truenet_data_postprocessing.get_final_3dvolumes(probs, test_sub_dict)
                    if verbose:
                        print(f'Saving the intermediate {plane} prediction ...', flush=True)

                    newhdr = flair_hdr.copy()
                    newobj = nib.nifti1.Nifti1Image(preds, None, header=newhdr)
                    nib.save(newobj, save_path)

            probs_combined = np.array(probs_combined)
            prob_mean = np.mean(probs_combined,axis = 0)

            save_path = truenet_utils.addSuffix(f'{output_dir}/Predicted_probmap_truenet_{basename}')
            pred_mean = truenet_data_postprocessing.get_final_3dvolumes(prob_mean, test_sub_dict)
            if verbose:
                print('Saving the final prediction ...', flush=True)

            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(pred_mean, None, header=newhdr)
            nib.save(newobj, save_path)

        if verbose:
            print('Fold ' + str(fld+1) + ': complete!', flush=True)

    if verbose:
        print('Cross-validation done!', flush=True)
