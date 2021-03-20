from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import nibabel as nib
from truenet.true_net import (truenet_loss_functions,
                              truenet_model, truenet_train, truenet_evaluate,
                              truenet_data_postprocessing)
from truenet.utils import (truenet_utils)

#=========================================================================================
# Truenet leave-one-evaluate function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def main(sub_name_dicts, loo_params, aug=True, weighted=True, intermediate=False,
         save_cp=True, save_wei=True, save_case='best', verbose=True, dir_cp=None, output_dir=None):
    '''
    The main function for leave-one-out validation of Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param loo_params: dictionary of LOO paramaters
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
    nclass = loo_params['Nclass']

    model_axial = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='axial')
    model_sagittal = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='sagittal')
    model_coronal = truenet_model.TrUENet(n_channels=2, n_classes=nclass, init_channels=64, plane='coronal')

    model_axial.to(device=device)
    model_sagittal.to(device=device)
    model_coronal.to(device=device)
    model_axial = nn.DataParallel(model_axial)
    model_sagittal = nn.DataParallel(model_sagittal)
    model_coronal = nn.DataParallel(model_coronal)

    optim_type = loo_params['Optimizer'] # adam, sgd
    milestones = loo_params['LR_Milestones']# list of integers [1, N]
    gamma = loo_params['LR_red_factor'] # scalar (0,1)
    lrt = loo_params['Learning_rate'] # scalar (0,1)
    train_prop = loo_params['Train_prop'] # scale (0,1)

    if type(milestones) != list:
        milestones = [milestones]

    if optim_type == 'adam':
        epsilon = loo_params['Epsilon']
        optimizer_axial = optim.Adam(model_axial.parameters(), lr=lrt, eps=epsilon)
        optimizer_sagittal = optim.Adam(model_sagittal.parameters(), lr=lrt, eps=epsilon)
        optimizer_coronal = optim.Adam(model_coronal.parameters(), lr=lrt, eps=epsilon)
    elif optim_type == 'sgd':
        moment = loo_params['Momentum']
        optimizer_axial = optim.SGD(model_axial.parameters(), lr=lrt, momentum=moment)
        optimizer_sagittal = optim.SGD(model_sagittal.parameters(), lr=lrt, momentum=moment)
        optimizer_coronal = optim.SGD(model_coronal.parameters(), lr=lrt, momentum=moment)
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    criterion = truenet_loss_functions.CombinedLoss()

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)
    for sub in range(len(sub_name_dicts)):
        if verbose:
            print('Training models for subject ' + str(sub+1) + '...', flush=True)
        test_sub_dict = [sub_name_dicts[sub]]
        basename = test_sub_dict[0]['basename']
        rem_sub_ids = np.setdiff1d(np.arange(len(sub_name_dicts)),sub)
        rem_sub_name_dicts = [sub_name_dicts[id] for id in rem_sub_ids]
        num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)),1)
        train_name_dicts, val_name_dicts, val_ids = truenet_utils.select_train_val_names(rem_sub_name_dicts,
                                                                                           num_val_subs)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_axial, milestones, gamma=gamma, last_epoch=-1)
        model_axial = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_axial, criterion,
                                                  optimizer_axial, scheduler, loo_params, device, mode='axial',
                                                  augment=aug, weighted=weighted, save_checkpoint=save_cp,
                                                  save_weights=save_wei, save_case=save_case, verbose=verbose,
                                                  dir_checkpoint=dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sagittal, milestones, gamma=gamma, last_epoch=-1)
        model_sagittal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_axial, criterion,
                                                  optimizer_sagittal, scheduler, loo_params, device, mode='sagittal',
                                                  augment=aug, weighted=weighted, save_checkpoint=save_cp,
                                                  save_weights=save_wei, save_case=save_case, verbose=verbose,
                                                  dir_checkpoint=dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_coronal, milestones, gamma=gamma, last_epoch=-1)
        model_coronal = truenet_train.train_truenet(train_name_dicts, val_name_dicts, model_axial, criterion,
                                                  optimizer_coronal, scheduler, loo_params, device, mode='coronal',
                                                  augment=aug, weighted=weighted, save_checkpoint=save_cp,
                                                  save_weights=save_wei, save_case=save_case, verbose=verbose,
                                                  dir_checkpoint=dir_cp)

        if verbose:
            print('Predicting output for subject ' + str(sub+1) + '...', flush=True)

        probs_combined = []
        flair_path = test_sub_dict[0]['flair_path']
        flair_hdr = nib.load(flair_path).header
        probs_axial = truenet_evaluate.evaluate_truenet(test_sub_dict, model_axial, loo_params, device,
                                                        mode='axial', verbose=verbose)
        probs_axial = truenet_data_postprocessing.resize_to_original_size(probs_axial, test_sub_dict,
                                                                          plane='axial')
        probs_combined.append(probs_axial)

        if intermediate:
            save_path = os.path.join(output_dir,'Predicted_probmap_truenet_' + basename + '_axial.nii.gz')
            preds_axial = truenet_data_postprocessing.get_final_3dvolumes(probs_axial, test_sub_dict)
            if verbose:
                print('Saving the intermediate Axial prediction ...', flush=True)

            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_axial, None, header=newhdr)
            nib.save(newobj, save_path)

        probs_sagittal = truenet_evaluate.evaluate_truenet(test_sub_dict, model_sagittal, loo_params, device,
                                                        mode='sagittal', verbose=verbose)
        probs_sagittal = truenet_data_postprocessing.resize_to_original_size(probs_sagittal, test_sub_dict,
                                                                          plane='sagittal')
        probs_combined.append(probs_sagittal)

        if intermediate:
            save_path = os.path.join(output_dir,'Predicted_probmap_truenet_' + basename + '_sagittal.nii.gz')
            preds_sagittal = truenet_data_postprocessing.get_final_3dvolumes(probs_sagittal, test_sub_dict)
            if verbose:
                print('Saving the intermediate Sagittal prediction ...', flush=True)

            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_sagittal, None, header=newhdr)
            nib.save(newobj, save_path)

        probs_coronal = truenet_evaluate.evaluate_truenet(test_sub_dict, model_coronal, loo_params, device,
                                                        mode='coronal', verbose=verbose)
        probs_coronal = truenet_data_postprocessing.resize_to_original_size(probs_coronal, test_sub_dict,
                                                                          plane='coronal')
        probs_combined.append(probs_coronal)

        if intermediate:
            save_path = os.path.join(output_dir,'Predicted_probmap_truenet_' + basename + '_coronal.nii.gz')
            preds_coronal = truenet_data_postprocessing.get_final_3dvolumes(probs_coronal, test_sub_dict)
            if verbose:
                print('Saving the intermediate Coronal prediction ...', flush=True)

            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_coronal, None, header=newhdr)
            nib.save(newobj, save_path)

        probs_combined = np.array(probs_combined)
        prob_mean = np.mean(probs_combined,axis = 0)

        save_path = os.path.join(output_dir,'Predicted_probmap_truenet_' + basename + '.nii.gz')
        pred_mean = truenet_data_postprocessing.get_final_3dvolumes(prob_mean, test_sub_dict)
        if verbose:
            print('Saving the final prediction ...', flush=True)

        newhdr = flair_hdr.copy()
        newobj = nib.nifti1.Nifti1Image(pred_mean, None, header=newhdr)
        nib.save(newobj, save_path)

    if verbose:
        print('Leave-one-out validation done!', flush=True)
