from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from truenet.true_net import (truenet_train_function, truenet_test_function,
                              truenet_cross_validate, truenet_finetune)
import glob

#=========================================================================================
# Truenet commands function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define the train sub-command for truenet
##########################################################################################


def find_model(args):
    """Prepares the path to the model files. """

    # Dictionary of pre-trained models: { <model-id> : <model-name> }.
    # The <model-id> is passed on the command-line, and is also the
    # directory name in $FSLDIR/data/truenet/models/
    pretrained_models = {
        'mwsc_flair' : 'Truenet_MWSC_FLAIR',
        'mwsc_t1'    : 'Truenet_MWSC_T1',
        'mwsc'       : 'Truenet_MWSC_FLAIR_T1',
        'mwsc_flair' : 'Truenet_MWSC_FLAIR',
        'mwsc_t1'    : 'Truenet_MWSC_T1',
        'mwsc'       : 'Truenet_MWSC_FLAIR_T1',
    }

    pretrained = args.model_name in pretrained_models

    if pretrained:
        pretrained = True
        model_dir = op.expandvars(f'$FSLDIR/data/truenet/models/{args.model_name}')
        if not op.exists(model_dir):
            model_dir = os.environ.get('TRUENET_PRETRAINED_MODEL_PATH', None)
            if model_dir is None:
                raise RuntimeError('Cannot find data; export TRUENET_PRETRAINED_'
                                   f'MODEL_PATH=/path/to/my/mwsc/{args.model_name}')

    # should be a file name prefix
    else:
        model_dir  = op.dirname(args.model_name)
        model_name = op.basename(args.model_name)
        axial      = f'{args.model_name}_axial.pth'
        sagittal   = f'{args.model_name}_sagittal.pth'
        coronal    = f'{args.model_name}_coronal.pth'
        if not (op.isfile(axial),
                op.isfile(sagittal),
                op.isfile(coronal)):
            raise ValueError(f'In directory {model_dir}, {axial} or {sagittal} or '
                             f'{coronal} does not appear to be a valid model file')

    return model_dir, model_name, pretrained


def train(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do basic sanity checks and assign variable names
    inp_dir = args.inp_dir

    flairflag = 0
    t1flag = 0

    input_flair_paths = glob.glob(os.path.join(inp_dir,'*_FLAIR.nii')) + \
              glob.glob(os.path.join(inp_dir,'*_FLAIR.nii.gz'))
    flairflag = 1

    input_t1_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                  glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))
    t1flag = 1

    if flairflag == 0 and t1flag == 0:
        raise ValueError(inp_dir + ' does not contain any FLAIR/T1 images / filenames NOT in required format')

    if flairflag == 1:
        input_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                            glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    else:
        input_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                         glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))

    model_dir = args.model_dir

    label_dir = args.label_dir

    if args.loss_function == 'weighted':
        gmdist_dir = args.gmdist_dir
        ventdist_dir = args.ventdist_dir
    else:
        gmdist_dir = None
        ventdist_dir = None

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    t1_count = 0
    flair_count = 0
    for l in range(len(input_paths)):
        flair_path_name = None
        t1_path_name = None
        if flairflag == 1:
            basepath = input_paths[l].split("_FLAIR.nii")[0]
            basename = basepath.split(os.sep)[-1]
            flair_path_name = input_paths[l]
            flair_count += 1
            print('FLAIR image found for ' + basename, flush=True)
            if os.path.isfile(basepath + '_T1.nii.gz'):
                t1_path_name = basepath + '_T1.nii.gz'
                t1_count += 1
            elif os.path.isfile(basepath + '_T1.nii'):
                t1_path_name = basepath + '_T1.nii'
                t1_count += 1
            else:
                print('T1 image not found for ' + basename + ', continuing...', flush=True)
        else:
            basepath = input_paths[l].split("_T1.nii")[0]
            basename = basepath.split(os.sep)[-1]
            t1_path_name = input_paths[l]
            print('T1 image found for ' + basename, flush=True)
            print('FLAIR image not found for ' + basename + ', continuing...', flush=True)

        # if os.path.isfile(basepath + '_T1.nii.gz'):
        #     t1_path_name = basepath + '_T1.nii.gz'
        # elif os.path.isfile(basepath + '_T1.nii'):
        #     t1_path_name = basepath + '_T1.nii'
        # else:
        #     raise ValueError('T1 file does not exist for ' + basename)

        print(os.path.join(label_dir, basename + '_manualmask.nii.gz'), flush=True)
        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        if args.loss_function == 'weighted':
            weighted = True
            if os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')
            elif os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii')
            else:
                raise ValueError('GM distance file does not exist for ' + basename)

            if os.path.isfile(os.path.join(ventdist_dir,basename + '_ventdistmap.nii.gz')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii.gz')
            elif os.path.isfile(os.path.join(ventdist_dir,basename + '_ventdistmap.nii')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii')
            else:
                raise ValueError('Ventricle distance file does not exist for ' + basename)
        else:
            weighted = False
            gmdist_path_name = None
            ventdist_path_name = None

        subj_name_dict = {'flair_path': flair_path_name,
                          't1_path': t1_path_name,
                          'gt_path': gt_path_name,
                          'gmdist_path': gmdist_path_name,
                          'ventdist_path': ventdist_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if flair_count > 0:
        if 0 < t1_count < flair_count or t1_count > flair_count:
            raise ValueError('For One or more subjects, T1 files are missing for corresponding FLAIR files')
        elif t1_count == 0:
            num_channels = 1
        elif t1_count == flair_count:
            num_channels = 2
    else:
        if t1_count > 0:
            num_channels = 1

    # Create the training parameters dictionary
    training_params = {'Learning_rate': args.init_learng_rate,
                       'Optimizer': args.optimizer,
                       'Epsilon' :args.epsilon,
                       'Momentum' : args.momentum,
                       'LR_Milestones': args.lr_sch_mlstone,
                       'LR_red_factor': args.lr_sch_gamma,
                       'Acq_plane': args.acq_plane,
                       'Train_prop': args.train_prop,
                       'Batch_size': args.batch_size,
                       'Num_epochs': args.num_epochs,
                       'Batch_factor': args.batch_factor,
                       'Patience': args.early_stop_val,
                       'Aug_factor': args.aug_factor,
                       'EveryN': args.cp_everyn_N,
                       'Nclass': args.num_classes,
                       'SaveResume': args.save_resume_training,
                       'Numchannels': num_channels
                       }

    save_wei = not args.save_full_model

    # Training main function call
    models = truenet_train_function.main(subj_name_dicts, training_params, aug=args.data_augmentation, weighted=weighted,
         save_cp=True, save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose, dir_cp=model_dir)


##########################################################################################
# Define the evaluate sub-command for truenet
##########################################################################################

def evaluate(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do basic sanity checks and assign variable names
    inp_dir = args.inp_dir
    out_dir = args.output_dir

    flairflag = 0
    t1flag = 0

    input_flair_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                        glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    flairflag = 1

    input_t1_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                     glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))
    t1flag = 1

    if flairflag == 0 and t1flag == 0:
        raise ValueError(inp_dir + ' does not contain any FLAIR/T1 images / filenames NOT in required format')

    if flairflag == 1:
        input_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    else:
        input_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))

    # Create a list of dictionaries containing required filepaths for the test subjects
    subj_name_dicts = []
    t1_count = 0
    flair_count = 0
    for l in range(len(input_paths)):
        flair_path_name = None
        t1_path_name = None
        if flairflag == 1:
            basepath = input_paths[l].split("_FLAIR.nii")[0]
            basename = basepath.split(os.sep)[-1]
            flair_path_name = input_paths[l]
            flair_count += 1
            print('FLAIR image found for ' + basename, flush=True)
            if os.path.isfile(basepath + '_T1.nii.gz'):
                t1_path_name = basepath + '_T1.nii.gz'
                t1_count += 1
            elif os.path.isfile(basepath + '_T1.nii'):
                t1_path_name = basepath + '_T1.nii'
                t1_count += 1
            else:
                print('T1 image not found for ' + basename + ', continuing...', flush=True)
        else:
            basepath = input_paths[l].split("_T1.nii")[0]
            basename = basepath.split(os.sep)[-1]
            t1_path_name = input_paths[l]
            print('T1 image found for ' + basename, flush=True)
            print('FLAIR image not found for ' + basename + ', continuing...', flush=True)

        # if os.path.isfile(basepath + '_T1.nii.gz'):
        #     t1_path_name = basepath + '_T1.nii.gz'
        # elif os.path.isfile(basepath + '_T1.nii'):
        #     t1_path_name = basepath + '_T1.nii'
        # else:
        #     raise ValueError('T1 file does not exist for ' + basename)

        subj_name_dict = {'flair_path': flair_path_name,
                          't1_path': t1_path_name,
                          'gmdist_path': None,
                          'ventdist_path': None,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if flair_count > 0:
        if 0 < t1_count < flair_count or t1_count > flair_count:
            raise ValueError('For One or more subjects, T1 files are missing for corresponding FLAIR files')
        elif t1_count == 0:
            num_channels = 1
        elif t1_count == flair_count:
            num_channels = 2
    else:
        if t1_count > 0:
            num_channels = 1

    model_dir, model_name, pretrained = find_model(args)

    # Create the training parameters dictionary
    eval_params = {'Nclass': args.num_classes,
                   'EveryN': args.cp_everyn_N,
                   'Pretrained': pretrained,
                   'Modelname': model_name,
                   'Numchannels': num_channels,
                   'Use_CPU': args.use_cpu
                   }

    # Test main function call
    truenet_test_function.main(subj_name_dicts, eval_params, intermediate=args.intermediate,
                               model_dir=model_dir, load_case=args.cp_load_type, output_dir=out_dir,
                               verbose=args.verbose)


##########################################################################################
# Define the fine_tune sub-command for truenet
##########################################################################################

def fine_tune(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do the usual sanity checks
    inp_dir = args.inp_dir

    flairflag = 0
    t1flag = 0

    input_flair_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                        glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    flairflag = 1

    input_t1_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                     glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))
    t1flag = 1
j
    if flairflag == 0 and t1flag == 0:
        raise ValueError(inp_dir + ' does not contain any FLAIR/T1 images / filenames NOT in required format')

    if flairflag == 1:
        input_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    else:
        input_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))

    out_dir = args.output_dir

    label_dir = args.label_dir

    if args.loss_function == 'weighted':
        gmdist_dir = args.gmdist_dir
        ventdist_dir = args.ventdist_dir
    else:
        gmdist_dir = None
        ventdist_dir = None

    # Create a list of dictionaries containing required filepaths for the fine-tuning subjects
    subj_name_dicts = []
    t1_count = 0
    flair_count = 0
    for l in range(len(input_paths)):
        flair_path_name = None
        t1_path_name = None
        if flairflag == 1:
            basepath = input_paths[l].split("_FLAIR.nii")[0]
            basename = basepath.split(os.sep)[-1]
            flair_path_name = input_paths[l]
            flair_count += 1
            print('FLAIR image found for ' + basename, flush=True)
            if os.path.isfile(basepath + '_T1.nii.gz'):
                t1_path_name = basepath + '_T1.nii.gz'
                t1_count += 1
            elif os.path.isfile(basepath + '_T1.nii'):
                t1_path_name = basepath + '_T1.nii'
                t1_count += 1
            else:
                print('T1 image not found for ' + basename + ', continuing...', flush=True)
        else:
            basepath = input_paths[l].split("_T1.nii")[0]
            basename = basepath.split(os.sep)[-1]
            t1_path_name = input_paths[l]
            print('T1 image found for ' + basename, flush=True)
            print('FLAIR image not found for ' + basename + ', continuing...', flush=True)

        # if os.path.isfile(basepath + '_T1.nii.gz'):
        #     t1_path_name = basepath + '_T1.nii.gz'
        # elif os.path.isfile(basepath + '_T1.nii'):
        #     t1_path_name = basepath + '_T1.nii'
        # else:
        #     raise ValueError('T1 file does not exist for ' + basename)

        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        if args.loss_function == 'weighted':
            weighted = True
            if os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')
            elif os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii')
            else:
                raise ValueError('GM distance file does not exist for ' + basename)

            if os.path.isfile(os.path.join(ventdist_dir, basename + '_ventdistmap.nii.gz')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii.gz')
            elif os.path.isfile(os.path.join(ventdist_dir, basename + '_ventdistmap.nii')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii')
            else:
                raise ValueError('Ventricle distance file does not exist for ' + basename)
        else:
            weighted = False
            gmdist_path_name = None
            ventdist_path_name = None

        subj_name_dict = {'flair_path': flair_path_name,
                          't1_path': t1_path_name,
                          'gt_path':gt_path_name,
                          'gmdist_path': gmdist_path_name,
                          'ventdist_path': ventdist_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if flair_count > 0:
        if 0 < t1_count < flair_count or t1_count > flair_count:
            raise ValueError('For One or more subjects, T1 files are missing for corresponding FLAIR files')
        elif t1_count == 0:
            num_channels = 1
        elif t1_count == flair_count:
            num_channels = 2
    else:
        if t1_count > 0:
            num_channels = 1

    if args.save_full_model == 'True':
        save_wei = False
    else:
        save_wei = True

    model_dir, model_name, pretrained = find_model(args)

    # Create the fine-tuning parameters dictionary
    finetuning_params = {'Finetuning_learning_rate': args.init_learng_rate,
                         'Optimizer': args.optimizer,
                         'Epsilon' :args.epsilon,
                         'Momentum' : args.momentum,
                         'LR_Milestones': args.lr_sch_mlstone,
                         'LR_red_factor': args.lr_sch_gamma,
                         'Acq_plane': args.acq_plane,
                         'Train_prop': args.train_prop,
                         'Batch_size': args.batch_size,
                         'Num_epochs': args.num_epochs,
                         'Batch_factor': args.batch_factor,
                         'Patience': args.early_stop_val,
                         'Aug_factor': args.aug_factor,
                         'EveryN': args.cp_everyn_N,
                         'Nclass': args.num_classes,
                         'Finetuning_layers': args.ft_layers,
                         'Load_type': args.cp_load_type,
                         'EveryNload': args.cpload_everyn_N,
                         'Pretrained': pretrained,
                         'Modelname': model_name,
                         'SaveResume': args.save_resume_training,
                         'Numchannels': num_channels,
                         'Use_CPU': args.use_cpu,
                         }

    # Fine-tuning main function call
    truenet_finetune.main(subj_name_dicts, finetuning_params, aug=args.data_augmentation, weighted=weighted,
                          save_cp=True, save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
                          model_dir=model_dir, dir_cp=out_dir)

##########################################################################################
# Define the loo_validate (leave-one-out validation) sub-command for truenet
##########################################################################################

def cross_validate(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Usual sanity check for checking if filepaths and files exist.
    inp_dir = args.inp_dir

    flairflag = 0
    t1flag = 0

    input_flair_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                        glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    flairflag = 1

    input_t1_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                     glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))
    t1flag = 1

    if flairflag == 0 and t1flag == 0:
        raise ValueError(inp_dir + ' does not contain any FLAIR/T1 images / filenames NOT in required format')

    if flairflag == 1:
        input_paths = glob.glob(os.path.join(inp_dir, '*_FLAIR.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_FLAIR.nii.gz'))
    else:
        input_paths = glob.glob(os.path.join(inp_dir, '*_T1.nii')) + \
                      glob.glob(os.path.join(inp_dir, '*_T1.nii.gz'))

    out_dir = args.output_dir
    label_dir = args.label_dir

    if args.loss_function == 'weighted':
        gmdist_dir = args.gmdist_dir
        ventdist_dir = args.ventdist_dir
    else:
        gmdist_dir = None
        ventdist_dir = None

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    t1_count = 0
    flair_count = 0
    for l in range(len(input_paths)):
        flair_path_name = None
        t1_path_name = None
        if flairflag == 1:
            basepath = input_paths[l].split("_FLAIR.nii")[0]
            basename = basepath.split(os.sep)[-1]
            flair_path_name = input_paths[l]
            flair_count += 1
            print('FLAIR image found for ' + basename, flush=True)
            if os.path.isfile(basepath + '_T1.nii.gz'):
                t1_path_name = basepath + '_T1.nii.gz'
                t1_count += 1
            elif os.path.isfile(basepath + '_T1.nii'):
                t1_path_name = basepath + '_T1.nii'
                t1_count += 1
            else:
                print('T1 image not found for ' + basename + ', continuing...', flush=True)
        else:
            basepath = input_paths[l].split("_T1.nii")[0]
            basename = basepath.split(os.sep)[-1]
            t1_path_name = input_paths[l]
            print('T1 image found for ' + basename, flush=True)
            print('FLAIR image not found for ' + basename + ', continuing...', flush=True)

        # if os.path.isfile(basepath + '_T1.nii.gz'):
        #     t1_path_name = basepath + '_T1.nii.gz'
        # elif os.path.isfile(basepath + '_T1.nii'):
        #     t1_path_name = basepath + '_T1.nii'
        # else:
        #     raise ValueError('T1 file does not exist for ' + basename)

        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        if args.loss_function == 'weighted':
            weighted = True
            if os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii.gz')
            elif os.path.isfile(os.path.join(gmdist_dir, basename + '_GMdistmap.nii')):
                gmdist_path_name = os.path.join(gmdist_dir, basename + '_GMdistmap.nii')
            else:
                raise ValueError('GM distance file does not exist for ' + basename)

            if os.path.isfile(os.path.join(ventdist_dir, basename + '_ventdistmap.nii.gz')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii.gz')
            elif os.path.isfile(os.path.join(ventdist_dir, basename + '_ventdistmap.nii')):
                ventdist_path_name = os.path.join(ventdist_dir, basename + '_ventdistmap.nii')
            else:
                raise ValueError('Ventricle distance file does not exist for ' + basename)
        else:
            weighted = False
            gmdist_path_name = None
            ventdist_path_name = None

        subj_name_dict = {'flair_path': flair_path_name,
                          't1_path': t1_path_name,
                          'gt_path': gt_path_name,
                          'gmdist_path': gmdist_path_name,
                          'ventdist_path': ventdist_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if flair_count > 0:
        if 0 < t1_count < flair_count or t1_count > flair_count:
            raise ValueError('For One or more subjects, T1 files are missing for corresponding FLAIR files')
        elif t1_count == 0:
            num_channels = 1
        elif t1_count == flair_count:
            num_channels = 2
    else:
        if t1_count > 0:
            num_channels = 1

    if len(subj_name_dicts) < args.cv_fold:
        raise ValueError('Number of folds is greater than number of subjects!')

    if args.resume_from_fold > args.cv_fold:
        raise ValueError('The fold to resume CV cannot be higher than the total number of folds specified!')

    # Create the loo_validate parameters dictionary
    cv_params = {'Learning_rate': args.init_learng_rate,
                 'fold': args.cv_fold,
                 'res_fold': args.resume_from_fold,
                 'Optimizer': args.optimizer,
                 'Epsilon':args.epsilon,
                 'Momentum': args.momentum,
                 'LR_Milestones': args.lr_sch_mlstone,
                 'LR_red_factor': args.lr_sch_gamma,
                 'Acq_plane': args.acq_plane,
                 'Train_prop': args.train_prop,
                 'Batch_size': args.batch_size,
                 'Num_epochs': args.num_epochs,
                 'Batch_factor': args.batch_factor,
                 'Patience': args.early_stop_val,
                 'Aug_factor': args.aug_factor,
                 'Nclass': args.num_classes,
                 'EveryN': args.cp_everyn_N,
                 'SaveResume': args.save_resume_training,
                 'Numchannels': num_channels
                 }

    save_wei = not args.save_full_model

    # Cross-validation main function call
    truenet_cross_validate.main(subj_name_dicts, cv_params, aug=args.data_augmentation, weighted=weighted,
                                intermediate=args.intermediate, save_cp=args.save_checkpoint, save_wei=save_wei,
                                save_case=args.cp_save_type, verbose=args.verbose, dir_cp=out_dir,
                                output_dir=out_dir)
