import os
import os.path as op
from truenet.true_net import (truenet_train_function, truenet_test_function,
                              truenet_cross_validate, truenet_finetune)

from fsl.scripts.imglob import imglob
from fsl.scripts.imtest import imtest
from fsl.data.image     import addExt

#=========================================================================================
# Truenet commands function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define the train sub-command for truenet
##########################################################################################


def gather_inputs(args, training):
    """Gathers paths to input files. """

    inp_dir  = args.inp_dir
    label_dir = None
    gmdist_dir = None
    ventdist_dir = None

    if training:
        label_dir = args.label_dir
        if args.loss_function == 'weighted':
            gmdist_dir   = args.gmdist_dir
            ventdist_dir = args.ventdist_dir

    input_flair_paths = imglob([f'{inp_dir}/*_FLAIR'])
    input_t1_paths    = imglob([f'{inp_dir}/*_T1'])
    flairflag         = len(input_flair_paths) > 0
    t1flag            = len(input_t1_paths) > 0

    if not (flairflag or t1flag):
        raise ValueError(inp_dir + ' does not contain any FLAIR/T1 images / filenames NOT in required format')

    if flairflag:
        input_paths = input_flair_paths
    else:
        input_paths = input_t1_paths

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    for l in range(len(input_paths)):
        flair_path_name = None
        t1_path_name = None
        gmdist_path_name = None
        ventdist_path_name = None
        gt_path_name = None

        if flairflag:
            flair_path_name = addExt(input_paths[l])
            basepath        = input_paths[l].removesuffix("_FLAIR")
            basename        = op.basename(basepath)
            print('FLAIR image found for ' + basename, flush=True)
            if imtest(f'{basepath}_T1'):
                t1_path_name = addExt(f'{basepath}_T1')
                print('T1 image found for ' + basename, flush=True)
        else:
            t1_path_name = addExt(input_paths[l])
            basepath     = input_paths[l].removesuffix("_T1")
            basename     = op.basename(basepath)
            print('T1 image found for ' + basename, flush=True)

        if training:
            if imtest(f'{label_dir}/{basename}_manualmask'):
                gt_path_name = addExt(f'{label_dir}/{basename}_manualmask')
            else:
                raise ValueError('Manual lesion mask does not exist for ' + basename)

            if args.loss_function == 'weighted':
                if imtest(f'{gmdist_dir}/{basename}_GMdistmap'):
                    gmdist_path_name = addExt(f'{gmdist_dir}/{basename}_GMdistmap')
                else:
                    raise ValueError('GM distance file does not exist for ' + basename)

                if imtest(f'{ventdist_dir}/{basename}_ventdistmap'):
                    ventdist_path_name = addExt(f'{ventdist_dir}/{basename}_ventdistmap')
                else:
                    raise ValueError('Ventricle distance file does not exist for ' + basename)

        subj_name_dict = {'flair_path': flair_path_name,
                          't1_path': t1_path_name,
                          'gt_path': gt_path_name,
                          'gmdist_path': gmdist_path_name,
                          'ventdist_path': ventdist_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    nflairs = len(input_flair_paths)
    nt1s    = len(input_t1_paths)
    if nflairs > 0:
        if nflairs != nt1s:
            raise ValueError('For one or more subjects, T1 files are missing for corresponding FLAIR files')
        elif nt1s == 0:
            num_channels = 1
        else:
            num_channels = 2
    else:
        num_channels = 1

    return subj_name_dicts, num_channels


def find_model(args):
    """Figures out the path to the pre-trained/custom model files.
    Returns the model directory and model name (which is the model
    file prefix, i.e. <model_name>_axial.pth, etc.).
    """

    # Dictionary of pre-trained models: { <model-id> : <model-prefix> }.
    # The <model-id> is passed on the command-line, and is also the
    # directory name in $FSLDIR/data/truenet/models/, i.e. pre-trained
    # models are named
    #
    # $FSLDIR/data/truenet/models/<model-id>/<model-prefix>_axial.pth
    #
    # etc
    pretrained_models = {
        'mwsc_flair' : 'Truenet_MWSC_FLAIR',
        'mwsc_t1'    : 'Truenet_MWSC_T1',
        'mwsc'       : 'Truenet_MWSC_FLAIR_T1',
        'ukbb_flair' : 'Truenet_UKBB_FLAIR',
        'ukbb_t1'    : 'Truenet_UKBB_T1',
        'ukbb'       : 'Truenet_UKBB_FLAIR_T1',
    }

    # We're given either the ID of a pre-trained
    # model, or a directory/file_name_prefix.

    # name of pre-trained model
    if args.model_name in pretrained_models:
        model_id   = args.model_name
        model_name = pretrained_models[model_id]

        # Search for model directory - will either
        # be in $FSLDIR/<model_id>, or in
        # $TRUENET_PRETRAINED_MODEL_PATH/<model_id>
        model_dir  = None
        candidates = [op.expandvars('$FSLDIR/data/truenet/models/'),
                      os.environ.get('TRUENET_PRETRAINED_MODEL_PATH', None)]

        for candidate in candidates:
            if candidate is None:
                continue

            candidate = f'{candidate}/{model_id}'

            if op.isdir(candidate):
                model_dir = candidate
                break

    # or a file name prefix
    else:
        model_dir  = op.abspath(op.dirname(args.model_name))
        model_name = op.basename(args.model_name)

    if (model_dir is None) or (not op.isdir(model_dir)):
        raise RuntimeError(
            'Cannot find TRUENET model files at {model_dir}/{model_name} '
            'check that the path/model name is correct, that pre-trained '
            'TRUENET models are installed, and/or export TRUENET_PRETRAINED_'
            'MODEL_PATH=/path/to/my/truenet/models/')

    axial      = f'{model_dir}/{model_name}_axial.pth'
    sagittal   = f'{model_dir}/{model_name}_sagittal.pth'
    coronal    = f'{model_dir}/{model_name}_coronal.pth'
    for model_file in [axial, sagittal, coronal]:
        if not op.isfile(model_file):
            raise ValueError(f'{model_file} does not appear to be a '
                             'valid TRUENET model file')

    return model_dir, model_name


def train(args):
    '''
    :param args: Input arguments from argparse
    '''

    subj_name_dicts, num_channels = gather_inputs(args, True)

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
    weighted = args.loss_function == 'weighted'

    # Training main function call
    truenet_train_function.main(
        subj_name_dicts, training_params,
        aug=args.data_augmentation, weighted=weighted,
        save_cp=True, save_wei=save_wei, save_case=args.cp_save_type,
        verbose=args.verbose, dir_cp=args.model_dir)


##########################################################################################
# Define the evaluate sub-command for truenet
##########################################################################################

def evaluate(args):
    '''
    :param args: Input arguments from argparse
    '''
    subj_name_dicts, num_channels = gather_inputs(args, False)
    model_dir, model_name = find_model(args)

    # Create the training parameters dictionary
    eval_params = {'EveryN': args.cp_everyn_N,
                   'Modelname': model_name,
                   'Numchannels': num_channels,
                   'Use_CPU': args.use_cpu
                   }

    # Test main function call
    truenet_test_function.main(
        subj_name_dicts, eval_params, intermediate=args.intermediate,
        model_dir=model_dir, load_case=args.cp_load_type,
        output_dir=args.output_dir, verbose=args.verbose)


##########################################################################################
# Define the fine_tune sub-command for truenet
##########################################################################################

def fine_tune(args):
    '''
    :param args: Input arguments from argparse
    '''
    subj_name_dicts, num_channels = gather_inputs(args, True)
    model_dir, model_name = find_model(args)

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
                         'Finetuning_layers': args.ft_layers,
                         'Load_type': args.cp_load_type,
                         'EveryNload': args.cpload_everyn_N,
                         'Modelname': model_name,
                         'SaveResume': args.save_resume_training,
                         'Numchannels': num_channels,
                         'Use_CPU': args.use_cpu,
                         }

    save_wei = not args.save_full_model
    weighted = args.loss_function == 'weighted'

    # Fine-tuning main function call
    truenet_finetune.main(
        subj_name_dicts, finetuning_params, aug=args.data_augmentation, weighted=weighted,
        save_cp=True, save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
        model_dir=model_dir, dir_cp=args.output_dir)

##########################################################################################
# Define the loo_validate (leave-one-out validation) sub-command for truenet
##########################################################################################

def cross_validate(args):
    '''
    :param args: Input arguments from argparse
    '''

    subj_name_dicts, num_channels = gather_inputs(args, True)

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
    weighted = args.loss_function == 'weighted'

    # Cross-validation main function call
    truenet_cross_validate.main(
        subj_name_dicts, cv_params, aug=args.data_augmentation, weighted=weighted,
        intermediate=args.intermediate, save_cp=args.save_checkpoint, save_wei=save_wei,
        save_case=args.cp_save_type, verbose=args.verbose, dir_cp=args.output_dir,
        output_dir=args.output_dir)
