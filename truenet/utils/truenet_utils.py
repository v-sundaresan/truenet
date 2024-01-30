from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import os
import torch
from collections import OrderedDict

#=========================================================================================
# Truenet general utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def select_train_val_names(data_path,val_numbers):
    '''
    Select training and validation subjects randomly given th no. of validation subjects
    :param data_path: input filepaths
    :param val_numbers: int, number of validation subjects
    :return:
    '''
    val_ids = random.choices(list(np.arange(len(data_path))),k=val_numbers)
    train_ids = np.setdiff1d(np.arange(len(data_path)),val_ids)    
    data_path_train = [data_path[ind] for ind in train_ids]
    data_path_val = [data_path[ind] for ind in val_ids]
    return data_path_train,data_path_val,val_ids


def freeze_layer_for_finetuning(model, layer_to_ft, verbose=False):
    '''
    Unfreezing specific layers of the model for fine-tuning
    :param model: model
    :param layer_to_ft: list of ints, layers to fine-tune starting from the decoder end.
    :param verbose: bool, display debug messages
    :return: model after unfreezing only the required layers
    '''
    model_layer_names = ['outconv', 'up1', 'up2', 'up3', 'down3', 'down2', 'down1', 'convfirst']
    model_layers_tobe_ftd = []
    for layer_id in layer_to_ft:
            model_layers_tobe_ftd.append(model_layer_names[layer_id-1])
            
    for name, child in model.module.named_children():
        if name in model_layers_tobe_ftd:
            if verbose:
                print('Model parameters', flush=True)
                print(name + ' is unfrozen', flush=True)
            for param in child.parameters():
                param.requires_grad = True
        else:
            if verbose:
                print('Model parameters', flush=True)
                print(name + ' is frozen', flush=True)
            for param in child.parameters():
                param.requires_grad = False
                
    return model


def loading_model(model_name, model, device, mode='weights'):
    if mode == 'weights':
        if device == 'cpu':
            print('utils:device used:' + device)
            axial_state_dict = torch.load(model_name, map_location='cpu')
        else:
            print('utils:device used:' + device)
            axial_state_dict = torch.load(model_name)
    else:
        if device == 'cpu':
            print('utils:device used:' + device)
            ckpt = torch.load(model_name, map_location='cpu')
        else:
            print('utils:device used:' + device)
            ckpt = torch.load(model_name)
        axial_state_dict = ckpt['model_state_dict']

    new_axial_state_dict = OrderedDict()
    for key, value in axial_state_dict.items():
        if 'module.' in key[:7]:
            name = key  # remove `module.`
        else:
            name = 'module.' + key
        new_axial_state_dict[name] = value
    model.load_state_dict(new_axial_state_dict)
    return model


class EarlyStoppingModelCheckpointing:
    '''
    Early stopping stops the training if the validation loss doesnt improve after a given patience
    '''
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss, tr_prms,
                 weights=True, checkpoint=True, save_condition='best', model_path=None, plane='axial'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 tr_prms, weights, checkpoint, save_condition, model_path, plane)
        elif score < self.best_score:  # Here is the criteria for activation of early stopping counter.
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:  # When the counter reaches the patience value, early stopping flag is activated to stop the training.
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 tr_prms, weights, checkpoint, save_condition, model_path, plane)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, best_val_acc, model, epoch, optimizer, scheduler, loss,
                        tr_prms, weights, checkpoint, save_condition, PATH, plane):
        # Saving checkpoints
        if checkpoint:
            # Saves the model when the validation loss decreases
            if self.verbose:
                print('Validation loss increased; Saving model ...')
            if weights:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Truenet_model_weights_bestdice_' + plane + '.pth')
                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), save_path)
                elif save_condition == 'everyN':
                    N = tr_prms['EveryN']
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH,
                                                 'Truenet_model_weights_epoch' + str(epoch) + '_' + plane + '.pth')
                        torch.save(model.state_dict(), save_path)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Truenet_model_weights_beforeES_' + plane + '.pth')
                    torch.save(model.state_dict(), save_path)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
            else:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Truenet_model_bestdice_' + plane + '.pth')
                    if val_acc > best_val_acc:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                elif save_condition == 'everyN':
                    N = tr_prms['EveryN']
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH, 'Truenet_model_epoch' + str(epoch) + '_' + plane + '.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Truenet_model_beforeES_' + plane + '.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_stat_dict': scheduler.state_dict(),
                        'loss': loss
                    }, save_path)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
        else:
            if self.verbose:
                print('Validation loss increased; Exiting without saving the model ...')







