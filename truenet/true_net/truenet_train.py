from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from truenet.true_net import truenet_data_preparation
from truenet.utils import (truenet_dataset_utils, truenet_utils)

#=========================================================================================
# Truenet training and validation functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def dice_coeff(inp, tar):
    '''
    Calculating Dice similarity coefficient
    :param inp: Input tensor
    :param tar: Target tensor
    :return: Dice value (scalar)
    '''
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice.cpu()

def validate_truenet(val_dataloader, model, batch_size, device, criterion, weighted=True, verbose=False):
    '''

    :param val_dataloader: Dataloader object
    :param model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param criterion: loss function
    :param weighted: bool, whether to apply spatial weights in loss function
    :param verbose: bool, display debug messages
    '''
    model.eval()
    dice_values = 0
    val_batch_count = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, val_dict in enumerate(val_dataloader):
            val_batch_count += 1
            X = val_dict['input']
            y = val_dict['gt']

            if verbose:
                print('Validation dimensions.......................................')
                print(X.shape)
                print(y.shape)

            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.double)

            if list(X.size())[0] == batch_size:
                val_pred = model.forward(X)
                
                if verbose:
                    print('Validation mask dimensions........')
                    print(val_pred.size())

                if weighted:
                    pix_weights = val_dict['pixweights']
                    pix_weights = pix_weights.to(device=device, dtype=torch.float32)
                    loss = criterion(val_pred, y, weight=pix_weights)
                else:
                    loss = criterion(val_pred, y, weight=None)
                 
                running_val_loss += loss.item()
                softmax = nn.Softmax()
                probs = softmax(val_pred)
                probs_vector = probs.contiguous().view(-1,2)
                mask_vector = (probs_vector[:,1] > 0.5).double()

                target_vector = y.contiguous().view(-1)
                dice_val = dice_coeff(mask_vector, target_vector)
                
                dice_values += dice_val
    val_av_loss = (running_val_loss / val_batch_count)
    val_dice = (dice_values / val_batch_count)
    print('Validation set: Average loss: ',  val_av_loss, flush=True)
    print('Validation set: Average accuracy: ',  val_dice, flush=True)
    return val_av_loss, val_dice

def train_truenet(train_name_dicts, val_names_dicts, model, criterion, optimizer, scheduler, train_params, device, 
                  mode='axial', augment=True, weighted=True, save_checkpoint=True, save_weights=True, save_case='best',
                  verbose=True, dir_checkpoint=None):
    '''
    Truenet train function
    :param train_name_dicts: list of dictionaries containing training filepaths
    :param val_names_dicts: list of dictionaries containing validation filepaths
    :param model: model
    :param criterion: loss function
    :param optimizer: optimiser
    :param scheduler: learning rate scheduler
    :param train_params: dictionary of training parameters
    :param device: cpu() or cuda()
    :param mode: str, acquisition plane
    :param augment: bool, perform data sugmentation
    :param weighted: bool, apply spatial weights in loss function
    :param save_checkpoint: bool
    :param save_weights: bool, if False, whole model will be saved
    :param save_case: str, condition for saving CP
    :param verbose: bool, display debug messages
    :param dir_checkpoint: str, filepath for saving the model
    :return: trained model
    '''
    
    batch_size = train_params['Batch_size']
    num_epochs = train_params['Num_epochs']   
    batch_factor = train_params['Batch_factor'] 
    patience = train_params['Patience']
    aug_factor = train_params['Aug_factor']
    save_resume = train_params['SaveResume']
    
    early_stopping = truenet_utils.EarlyStoppingModelCheckpointing(patience, verbose=verbose)
        
    num_iters = max(len(train_name_dicts)//batch_factor,1)
    losses_train = []
    losses_val = []
    dice_val = []
    best_val_dice = 0
    
    val_data_dict = truenet_data_preparation.create_data_array(val_names_dicts, is_weighted=weighted, plane=mode)
    valdata = truenet_data_preparation.get_slices_from_data_with_aug(val_data_dict, plane=mode, test=1,
                                                                     weighted=weighted)

    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_' + mode + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_' + mode + '.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['loss_train']
            losses_val = checkpoint_resumetraining['loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']
        except:
            if verbose:
                print('Not found any model to load and resume training!', flush=True)

    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        running_loss = 0.0
        batch_count = 0   
        print('Epoch: ' + str(epoch) + ' starting!..............................')
        for i in range(num_iters):
            trainnames = train_name_dicts[i*batch_factor:(i+1)*batch_factor]
            print('Training files names listing...................................')
            print(trainnames)
            train_data_dict = truenet_data_preparation.create_data_array(trainnames, is_weighted=weighted, plane=mode)
            if augment:
                traindata = truenet_data_preparation.get_slices_from_data_with_aug(train_data_dict, af=aug_factor,
                                                                                   plane=mode, test=0, weighted=weighted)
            else:
                traindata = truenet_data_preparation.get_slices_from_data_with_aug(train_data_dict, plane=mode,
                                                                                   test=1, weighted=weighted)
            
            data = traindata[0].transpose(0,3,1,2)
            label = traindata[1]
            data_val = valdata[0].transpose(0,3,1,2)
            label_val = valdata[1]
            if weighted:
                pix_weights = (traindata[2] + traindata[3]) * (traindata[3] > 0).astype(float)
                pix_weights_val = (valdata[2] + valdata[3]) * (valdata[3] > 0).astype(float)
                train_dataset_dict = truenet_dataset_utils.WMHDatasetWeighted(data, label, pix_weights)
                val_dataset_dict = truenet_dataset_utils.WMHDatasetWeighted(data_val, label_val, pix_weights_val)
            else:
                train_dataset_dict = truenet_dataset_utils.WMHDataset(data, label)
                val_dataset_dict = truenet_dataset_utils.WMHDataset(data_val, label_val)

            train_dataloader = DataLoader(train_dataset_dict, batch_size=batch_size, shuffle=True, num_workers=0)
            val_dataloader = DataLoader(val_dataset_dict, batch_size=batch_size, shuffle=False, num_workers=0)

            for batch_idx, train_dict in enumerate(train_dataloader):
                model.train()
                X = train_dict['input']
                y = train_dict['gt']

                if verbose:
                    print('Training dimensions.......................................')
                    print(X.shape)
                    print(y.shape)
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.double)
                
                if list(X.size())[0] == batch_size :  
                    optimizer.zero_grad()
                    masks_pred = model.forward(X)

                    if verbose:
                        print('mask_pred dimensions........')
                        print(masks_pred.size())

                    if weighted:
                        pix_weights = train_dict['pixweights']
                        pix_weights = pix_weights.to(device=device, dtype=torch.float32)
                        loss = criterion(masks_pred, y, weight=pix_weights)
                    else:
                        loss = criterion(masks_pred, y, weight=None)

                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    if verbose:
                        if batch_idx % 10 == 0:
                            print('Train Mini-batch: {} out of Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                (i+1), epoch, (batch_idx+1) * len(X), len(train_dataloader.dataset),
                                100. * (batch_idx+1) / len(train_dataloader), loss.item()), flush=True)

                    batch_count += 1
                
        val_av_loss, val_av_dice = validate_truenet(val_dataloader, model, batch_size, device, criterion,
                                                    weighted=weighted, verbose=verbose)
        scheduler.step(val_av_dice)
                    
        av_loss = (running_loss / batch_count)#.detach().cpu().numpy()
        print('Training set: Average loss: ',  av_loss, flush=True)
        losses_train.append(av_loss)        
        losses_val.append(val_av_loss)
        dice_val.append(val_av_dice)

        if save_resume:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_' + mode + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_' + mode + '.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss_train': losses_train,
                'loss_val': losses_val,
                'dice_val': dice_val,
                'best_val_dice': best_val_dice
            }, ckpt_path)

        if save_checkpoint:
            np.savez(os.path.join(dir_checkpoint,'losses_' + mode + '.npz'), train_loss=losses_train, val_loss=losses_val)
            np.savez(os.path.join(dir_checkpoint,'validation_dice_' + mode + '.npz'), dice_val=dice_val[0].detach().cpu().numpy())
        
        early_stopping(val_av_loss, val_av_dice, best_val_dice, model, epoch, optimizer, scheduler, av_loss, 
                       train_params, weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case, 
                       model_path=dir_checkpoint, plane=mode)
        
        if val_av_dice > best_val_dice:
            best_val_dice = val_av_dice

        if early_stopping.early_stop: 
            print('Patience Reached - Early Stopping Activated', flush=True)
            if save_resume:
                if dir_checkpoint is not None:
                    ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_' + mode + '.pth')
                else:
                    ckpt_path = os.path.join(os.getcwd(), 'tmp_model_' + mode + '.pth')
                os.remove(ckpt_path)
            return model
#             sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    if save_resume:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_' + mode + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model_' + mode + '.pth')
        os.remove(ckpt_path)

    return model
        








