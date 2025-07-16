import os
import os.path as op
import glob
from truenet.true_net import (truenet_train_function, truenet_test_function,
                              truenet_cross_validate, truenet_finetune)

from fsl.scripts.imglob import imglob
from fsl.data.image     import addExt

#=========================================================================================
# Truenet commands function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define the train sub-command for truenet
##########################################################################################


def find_inputs(args, training):
    """Gathers paths to input files. Returns a list of dictionaries, one for each
    subject, containing input file paths.
    """

    ignore_flair = args.t1_only
    ignore_t1    = args.flair_only
    inp_dir      = args.inp_dir
    label_dir    = None
    gmdist_dir   = None
    ventdist_dir = None

    if training:
        label_dir = args.label_dir
        if args.loss_function == 'weighted':
            gmdist_dir   = args.gmdist_dir
            ventdist_dir = args.ventdist_dir

    flair_paths = imglob([f'{inp_dir}/*_FLAIR'])
    t1_paths    = imglob([f'{inp_dir}/*_T1'])
    have_flair  = (len(flair_paths)) > 0 and (not ignore_flair)
    have_t1     = (len(t1_paths))    > 0 and (not ignore_t1)

    if not (have_flair or have_t1):
        raise ValueError(f'Cannot find any FLAIR/T1 images in {inp_dir} - '
                         'check that the input directory is correct, and '
                         'that files are named appropriately, e.g. '
                         '<subj-id>_FLAIR.nii.gz')

    if have_flair:
        subj_ids = [op.basename(p.removesuffix('_FLAIR')) for p in flair_paths]
    else:
        subj_ids = [op.basename(p.removesuffix('_T1')) for p in t1_paths]

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    for subj_id in subj_ids:
        flair_path    = None
        t1_path       = None
        gmdist_path   = None
        ventdist_path = None
        gt_path       = None

        if have_flair and (not ignore_flair):
            try:
                flair_path = addExt(f'{inp_dir}/{subj_id}_FLAIR')
                print(f'FLAIR image found for {subj_id}')
            except Exception:
                raise ValueError(f'FLAIR image missing: {inp_dir}/{subj_id}_FLAIR')

        if have_t1 and (not ignore_t1):
            try:
                t1_path = addExt(f'{inp_dir}/{subj_id}_T1')
                print(f'T1 image found for {subj_id}')
            except Exception:
                raise ValueError(f'T1 image missing for {inp_dir}/{subj_id}_T1')

        if training:
            try:
                gt_path = addExt(f'{label_dir}/{subj_id}_manualmask')
                print(f'Lesion mask found for {subj_id}')
            except Exception:
                raise ValueError(f'Manual lesion mask missing: {label_dir}/{subj_id}_manualmask')

        if training and (args.loss_function == 'weighted'):
            try:
                gmdist_path = addExt(f'{gmdist_dir}/{subj_id}_GMdistmap')
                print(f'GM distance map found for {subj_id}')
            except Exception:
                raise ValueError(f'GM distance map missing: {gmdist_dir}/{subj_id}_GMdistmap')

            try:
                ventdist_path = addExt(f'{ventdist_dir}/{subj_id}_ventdistmap')
                print(f'Ventricle distance map found for {subj_id}')
            except Exception:
                raise ValueError(f'Ventricle distance map missing: {ventdist_dir}/{subj_id}_ventdistmap')

        subj_name_dicts.append({
            'flair_path'    : flair_path,
            't1_path'       : t1_path,
            'gt_path'       : gt_path,
            'gmdist_path'   : gmdist_path,
            'ventdist_path' : ventdist_path,
            'basename'      : subj_id
        })

    num_channels = int(have_flair) + int(have_t1)

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

    # or a directory/file name prefix
    else:

        # we've been given a directory
        if op.isdir(args.model_name):
            model_dir = args.model_name
            # identify the model file name prefix
            axfile = glob.glob(f'{model_dir}/*_axial.pth')

            # fall-through to error handling below
            if len(axfile) == 0:
                model_name = ''

            # if multiple models, just use the first one.
            # The user would have to specify which model
            # to use
            else:
                model_name = op.basename(axfile[0]).removesuffix('_axial.pth')

        # we've been given a filename prefix
        else:
            model_dir  = op.dirname( args.model_name)
            model_name = op.basename(args.model_name)

    axial      = f'{model_dir}/{model_name}_axial.pth'
    sagittal   = f'{model_dir}/{model_name}_sagittal.pth'
    coronal    = f'{model_dir}/{model_name}_coronal.pth'

    error_msg = ('Cannot find TRUENET model files at {model_dir}/{model_name} '
                 'check that the path/model name is correct, that pre-trained '
                 'TRUENET models are installed, and/or export TRUENET_'
                 'PRETRAINED_MODEL_PATH=/path/to/my/truenet/models/')

    if (model_dir is None) or (not op.isdir(model_dir)):
        raise RuntimeError(error_msg)
    for model_file in [axial, sagittal, coronal]:
        if not op.isfile(model_file):
            raise ValueError(error_msg)

    print(f'Found TRUENET model {model_dir}/{model_name}')

    return op.abspath(model_dir), model_name


def train(args):
    '''
    :param args: Input arguments from argparse
    '''

    subj_name_dicts, num_channels = find_inputs(args, True)

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
    subj_name_dicts, num_channels = find_inputs(args, False)
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
    subj_name_dicts, num_channels = find_inputs(args, True)
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

    subj_name_dicts, num_channels = find_inputs(args, True)

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
