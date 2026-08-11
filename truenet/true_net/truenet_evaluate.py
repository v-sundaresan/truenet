import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from truenet.true_net import truenet_data_preparation
from truenet.utils import truenet_dataset_utils

#=========================================================================================
# Truenet evaluate function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


def evaluate_truenet(test_name_dicts, model, test_params, device, mode='axial', verbose=False):
    '''
    Truenet evaluate function definition
    :param test_name_dicts: list of dictionaries with test filepaths
    :param model: test model
    :param test_params: parameters used for testing
    :param device: pytorch device
    :param mode: acquisition plane
    :param verbose: display debug messages
    :return: predicted probability array
    '''
    testdata = truenet_data_preparation.create_test_data_array(test_name_dicts, plane=mode)
    data = testdata[0].transpose(0,3,1,2)

    test_dataset_dict = truenet_dataset_utils.WMHTestDataset(data)
    test_dataloader = DataLoader(test_dataset_dict, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    prob_array = np.array([])
    with torch.no_grad():
        for batchidx, test_dict in enumerate(test_dataloader):
            X = test_dict['input']

            if verbose:
                print('Testdata dimensions.......................................')
                print(X.size())

            X = X.to(device=device, dtype=torch.float32)
            val_pred = model.forward(X)

            if verbose:
                print('Validation mask dimensions........')
                print(val_pred.size())

            softmax = nn.Softmax(dim=1)
            probs = softmax(val_pred)

            probs_nparray = probs.detach().cpu().numpy()

            prob_array = np.concatenate((prob_array,probs_nparray),axis=0) if prob_array.size else probs_nparray

    prob_array = prob_array.transpose(0,2,3,1)
    return prob_array
