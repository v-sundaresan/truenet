import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from truenet.true_net import (truenet_model, truenet_evaluate, truenet_data_postprocessing)
from truenet.utils import truenet_utils


#=========================================================================================
# Truenet main test function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

def main(sub_name_dicts, eval_params, intermediate=False, model_dir=None,
         load_case='last', output_dir=None, verbose=False):
    '''
    The main function for testing Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param eval_params: dictionary of evaluation parameters
    :param intermediate: bool, whether to save intermediate results
    :param model_dir: str, filepath containing the test model
    :param load_case: str, condition for loading the checkpoint
    :param output_dir: str, filepath for saving the output predictions
    :param verbose: bool, display debug messages
    '''
    assert len(sub_name_dicts) > 0, "There must be at least 1 subject for testing."

    if eval_params['Use_CPU']:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # number of channels (T1/FLAIR) present in input
    input_channels = eval_params['Numchannels']
    model_name     = eval_params['Modelname']

    # peek at one of the model files to identify
    # expected number of input channels and output
    # classes
    nclasses, nchannels = truenet_utils.peek_model(f'{model_dir}/{model_name}_axial.pth')

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

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    for sub in range(len(sub_name_dicts)):
        if verbose:
            print('Predicting output for subject ' + str(sub+1) + '...', flush=True)

        test_sub_dict = sub_name_dicts[sub]
        basename      = test_sub_dict['basename']

        probs_combined = []
        if test_sub_dict['flair_path'] is not None:
            input_path = test_sub_dict['flair_path']
        else:
            input_path = test_sub_dict['t1_path']

        input_hdr = nib.load(input_path).header

        for plane, model in models.items():
            probs = truenet_evaluate.evaluate_truenet(
                [test_sub_dict], model, eval_params, device,
                mode=plane, verbose=verbose)
            probs = truenet_data_postprocessing.resize_to_original_size(
                probs, input_path, plane=plane)
            probs_combined.append(probs)

            if intermediate:
                save_path = truenet_utils.addSuffix(f'{output_dir}/Predicted_probmap_truenet_{basename}_{plane}')
                preds = truenet_data_postprocessing.get_final_3dvolumes(probs, input_path)
                if verbose:
                    print(f'Saving the intermediate {plane} prediction ...', flush=True)

                newhdr = input_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(preds, None, header=newhdr)
                nib.save(newobj, save_path)

        probs_combined = np.array(probs_combined)
        prob_mean = np.mean(probs_combined,axis=0)

        save_path = truenet_utils.addSuffix(f'{output_dir}/Predicted_probmap_truenet_{basename}')
        pred_mean = truenet_data_postprocessing.get_final_3dvolumes(prob_mean, input_path)
        if verbose:
            print('Saving the final prediction ...', flush=True)

        newhdr = input_hdr.copy()
        newobj = nib.nifti1.Nifti1Image(pred_mean, None, header=newhdr)
        nib.save(newobj, save_path)

    if verbose:
        print('Testing complete for all subjects!', flush=True)
