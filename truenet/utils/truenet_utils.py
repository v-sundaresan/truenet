import numpy as np
import random
import os
import torch
import fnmatch

from collections import OrderedDict

from fsl.data.image import addExt


#=========================================================================================
# Truenet general utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================


def addSuffix(prefix):
    """Adds a .nii/.nii.gz suffix to the given file prefix, based on the
    value of the $FSLOUTPUTTYPE environment variable.
    """
    return addExt(prefix, mustExist=False)


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


def load_model(model_path, model, device):
    """Load a saved TRUENET model file. """

    # Models could have been saved with or without
    # --save_full_model - if this was true, the
    # model contains a bunch of other information
    # that we don't need - we only want the model
    # state dict, which will hanve been stored
    # under a key named "model_state_dict".
    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model.load_state_dict(state_dict)

    except Exception as e:
        raise Exception(f'{model_path} does not appear to be '
                        'a valid truenet model file') from e

    return model


def peek_model(model_path):
    """Peeks inside the given model file and returns the number of input
    channels and output classes.
    """

    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # see comment in load model above
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        nclasses  = None
        nchannels = None

        for key, value in state_dict.items():
            if fnmatch.fnmatch(key, '*.outconv.*.weight'):
                nclasses = value.size()[0]
            if fnmatch.fnmatch(key, '*.inpconv.*.weight'):
                nchannels = value.size()[1]

        return nclasses, nchannels

        if nclasses is None or nchannels is None:
            raise Exception()

    except Exception as e:
        raise Exception(f'{model_path} does not appear to be '
                        'a valid truenet model file') from e


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
