#!/usr/bin/env python

import argparse
import os.path as op
import importlib.metadata as impmeta
import sys

from truenet.true_net import truenet_commands

DESCRIPTIONS = {
    'train' :
    "The \'train\' command trains the TrUE-Net model from scratch using the "
    "training subjects specified in the input directory. The FLAIR and T1 "
    "volumes should be named as '<subj_name>_FLAIR.nii.gz' and '<subj_name>_"
    "T1.nii.gz' respectively.",

    'evaluate':
    "The 'evaluate' command is used for testing the TrUE-Net model on the "
    "test subjects specified in the input directory. The FLAIR and T1 volumes "
    "should be named as '<subj_name>_FLAIR.nii.gz' and '<subj_name>_"
    "T1.nii.gz' respectively",

    'fine_tune':
    "The 'fine_tune' command fine-tunes a pretrained TrUE-Net model (from a "
    "model directory) on the training subjects specified in the input "
    "directory. The FLAIR and T1 volumes should be named as '<subj_name>_"
    "FLAIR.nii.gz' and '<subj_name>_T1.nii.gz' respectively.",

    'cross_validate':
    "The 'cross_validate' command performs cross-validation of the TrUE-Net "
    "model on the subjects specified in the input directory. The FLAIR and "
    "T1 volumes should be named as '<subj_name>_FLAIR.nii.gz' and "
    "'<subj_name>_T1.nii.gz' respectively"
}

EPILOGS = {
    'mainparser' :
    'For detailed help regarding the options for each command, type '
    'truenet <command> --help or -h (e.g. truenet train --help, '
    'truenet train -h)',

    'subparsers' :
    "For detailed help regarding the options for each argument, refer to the "
    "user-guide or readme document. For more details on TrUE-Net, refer to "
    "https://www.biorxiv.org/content/10.1101/2020.07.24.219485v1.full"
}


#=========================================================================================
# FSL TRUE_NET
# Vaanathi Sundaresan
# 01-04-2021, Oxford
#=========================================================================================
def main():
    version = impmeta.version("truenet")
    parser = argparse.ArgumentParser(
        prog='truenet',
        description=f'truenet: Triplanar ensemble U-Net model v{version}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOGS['mainparser'])
    subparsers = parser.add_subparsers(
        title='Commands available',
        dest="command", required=True)

    parser_train = subparsers.add_parser(
        'train',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Training a TrUE-Net model from scratch',
        description=DESCRIPTIONS['train'],
        epilog=EPILOGS['subparsers'])
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Applying a saved/pretrained TrUE-Net model',
        description=DESCRIPTIONS['evaluate'],
        epilog=EPILOGS['subparsers'])
    parser_finetune = subparsers.add_parser(
        'fine_tune',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Fine-tuning a saved/pretrained TrUE-Net model',
        description=DESCRIPTIONS['fine_tune'],
        epilog=EPILOGS['subparsers'])
    parser_cv = subparsers.add_parser(
        'cross_validate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Cross-validation of TrUE-Net model',
        description=DESCRIPTIONS['cross_validate'],
        epilog=EPILOGS['subparsers'])

    requiredTrain = parser_train.add_argument_group('Required arguments')
    optionalTrain = parser_train.add_argument_group('Optional arguments')

    requiredTrain.add_argument('-i', '--inp_dir', required=True, help='Input directory containing training images')
    requiredTrain.add_argument('-l', '--label_dir', required=True, help='Directory containing lesion manual masks')
    requiredTrain.add_argument('-m', '--model_dir', required=True, help='Directory for saving model weights')

    optionalTrain.add_argument('-tr_prop', '--train_prop', type = float, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalTrain.add_argument('-bfactor', '--batch_factor', type = int, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalTrain.add_argument('-loss', '--loss_function', choices=('weighted', 'nweighted'), default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalTrain.add_argument('-gdir', '--gmdist_dir', default=None, help='Directory containing GM distance map images (default: --inp_dir).')
    optionalTrain.add_argument('-vdir', '--ventdist_dir', help='Directory containing ventricle distance map images. (default: --inp_dir).')
    optionalTrain.add_argument('-nclass', '--num_classes', type = int, default=2, help='No of classes to consider in the target labels; any additional class will be considered part of background (default=2)')
    optionalTrain.add_argument('-plane', '--acq_plane', choices=('all', 'axial', 'sagittal', 'coronal'), default='all', help='Options: axial, sagittal, coronal, all (default = all)')
    optionalTrain.add_argument('-da', '--data_augmentation', action='store_false', help='Applying data augmentation (default=True)')
    optionalTrain.add_argument('-af', '--aug_factor', type = int, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalTrain.add_argument('-sv_resume', '--save_resume_training', action='store_true', help='Whether to save and resume training in case of interruptions (default-False)')
    optionalTrain.add_argument('-ilr', '--init_learng_rate', type = float, default=0.001, help='Initial LR to use in scheduler (default=0.001)')
    optionalTrain.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, default=10, help='Milestones for LR scheduler (default=10)')
    optionalTrain.add_argument('-gamma', '--lr_sch_gamma', type = float, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalTrain.add_argument('-opt', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Optimizer used for training. Options: adam, sgd (default=adam)')
    optionalTrain.add_argument('-eps', '--epsilon', type = float, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalTrain.add_argument('-mom', '--momentum', type = float, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalTrain.add_argument('-bs', '--batch_size', type = int, default=8, help='Batch size (default=8)')
    optionalTrain.add_argument('-ep', '--num_epochs', type = int, default=60, help='Number of epochs (default=60)')
    optionalTrain.add_argument('-es', '--early_stop_val', type = int, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalTrain.add_argument('-sv_mod', '--save_full_model', action='store_true', help='Saving the whole model instead of weights alone (default=False)')
    optionalTrain.add_argument('-cp_type', '--cp_save_type', choices=('best', 'last', 'everyN'), default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalTrain.add_argument('-cp_n', '--cp_everyn_N', type = int, default=10, help='If -cp_type=everyN, the N value (default=10)')
    optionalTrain.add_argument('-to', '--t1_only', action='store_true', help='Only use T1 images (ignore FLAIR images if present)')
    optionalTrain.add_argument('-fo', '--flair_only', action='store_true', help='Only use FLAIR images (ignore T1 images if present)')
    optionalTrain.add_argument('-v', '--verbose', action='store_true', help='Display debug messages (default=False)')

    requiredEvaluate = parser_evaluate.add_argument_group('Required arguments')
    optionalEvaluate = parser_evaluate.add_argument_group('Optional arguments')

    requiredEvaluate.add_argument('-i', '--inp_dir', required=True, help='Input directory containing test images')
    requiredEvaluate.add_argument('-m', '--model_name', required=True, help='Pretrained model name or model file basename')
    requiredEvaluate.add_argument('-o', '--output_dir', required=True, help='Directory for saving predictions')

    optionalEvaluate.add_argument('-cpu', '--use_cpu', action='store_true', help='Perform model evaluation on CPU True/False (default=False)')
    optionalEvaluate.add_argument('-int', '--intermediate', action='store_true', help='Saving intermediate predictionss (individual planes) for each subject (default=False)')
    optionalEvaluate.add_argument('-cp_type', '--cp_load_type', default='last', help='Checkpoint to be loaded. Options: best, last, specific (default = last)')
    optionalEvaluate.add_argument('-cp_n', '--cp_everyn_N', type = int, help='If -cp_type=specific, the N value (default=10)')
    optionalEvaluate.add_argument('-to', '--t1_only', action='store_true', help='Only use T1 images (ignore FLAIR images if present)')
    optionalEvaluate.add_argument('-fo', '--flair_only', action='store_true', help='Only use FLAIR images (ignore T1 images if present)')
    optionalEvaluate.add_argument('-v', '--verbose', action='store_true', help='Display debug messages (default=False)')

    requiredFt = parser_finetune.add_argument_group('Required arguments')
    optionalFt = parser_finetune.add_argument_group('Optional arguments')

    requiredFt.add_argument('-i', '--inp_dir', required=True, help='Input directory containing training images')
    requiredFt.add_argument('-l', '--label_dir', required=True, help='Directory containing lesion manual masks')
    requiredFt.add_argument('-m', '--model_name', required=True, help='Pretrained model name or model file basename')
    requiredFt.add_argument('-o', '--output_dir', required=True, help='Output directory for saving fine-tuned models/weights')

    optionalFt.add_argument('-cpld_type', '--cp_load_type', choices=('best', 'last', 'specific'), default='last', help='Checkpoint to be loaded. Options: best, last, specific (default=last')
    optionalFt.add_argument('-cpld_n', '--cpload_everyn_N', type = int, default=10, help='If -cpld_type=specific, the N value (default=10)')
    optionalFt.add_argument('-ftlayers', '--ft_layers', nargs='+', type=int, default=2, help='Layers to fine-tune starting from the decoder (default=1 2)')
    optionalFt.add_argument('-tr_prop', '--train_prop', type = float, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalFt.add_argument('-bfactor', '--batch_factor', type = int, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalFt.add_argument('-loss', '--loss_function', choices=('weighted', 'nweighted'), default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalFt.add_argument('-gdir', '--gmdist_dir', help='Directory containing GM distance map images (default: --inp_dir).')
    optionalFt.add_argument('-vdir', '--ventdist_dir', help='Directory containing ventricle distance map images (default: --inp_dir).')
    optionalFt.add_argument('-plane', '--acq_plane', choices=('all', 'axial', 'sagittal', 'coronal'), default='all', help='The plane in which the model needs to be fine-tuned. Options: axial, sagittal, coronal, all (default=all)')
    optionalFt.add_argument('-da', '--data_augmentation', action='store_false', help='Applying data augmentation (default=True)')
    optionalFt.add_argument('-af', '--aug_factor', type = int, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalFt.add_argument('-sv_resume', '--save_resume_training', action='store_true', help='Whether to save and resume training in case of interruptions (default-False)')
    optionalFt.add_argument('-ilr', '--init_learng_rate', type = float, default=0.0001, help='Initial LR to use for fine-tuning in scheduler (default=0.0001)')
    optionalFt.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, default=10, help='Milestones for LR scheduler (default=10)')
    optionalFt.add_argument('-gamma', '--lr_sch_gamma', type = float, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalFt.add_argument('-opt', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Optimizer used for training. Options:adam, sgd (default=adam)')
    optionalFt.add_argument('-eps', '--epsilon', type = float, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalFt.add_argument('-mom', '--momentum', type = float, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalFt.add_argument('-bs', '--batch_size', type = int, default=8, help='Batch size (default=8)')
    optionalFt.add_argument('-ep', '--num_epochs', type = int, default=60, help='Number of epochs (default=60)')
    optionalFt.add_argument('-es', '--early_stop_val', type = int, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalFt.add_argument('-sv_mod', '--save_full_model', action='store_true', help='Saving the whole model instead of weights alone (default=False)')
    optionalFt.add_argument('-cp_type', '--cp_save_type', choices=('best', 'last', 'everyN'), default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalFt.add_argument('-cp_n', '--cp_everyn_N', type = int, default=10, help='If -cp_type=everyN, the N value')
    optionalFt.add_argument('-cpu', '--use_cpu', action='store_true', help='Perform model fine-tuning on CPU True/False (default=False)')
    optionalFt.add_argument('-to', '--t1_only', action='store_true', help='Only use T1 images (ignore FLAIR images if present)')
    optionalFt.add_argument('-fo', '--flair_only', action='store_true', help='Only use FLAIR images (ignore T1 images if present)')
    optionalFt.add_argument('-v', '--verbose', action='store_true', help='Display debug messages (default=False)')

    requiredCv = parser_cv.add_argument_group('Required arguments')
    optionalCv = parser_cv.add_argument_group('Optional arguments')

    requiredCv.add_argument('-i', '--inp_dir', required=True, help='Input directory containing images')
    requiredCv.add_argument('-l', '--label_dir', required=True, help='Directory containing lesion manual masks')
    requiredCv.add_argument('-o', '--output_dir', required=True, help='Output directory for saving predictions (and models)')

    optionalCv.add_argument('-fold', '--cv_fold', type = int, default=5, help='Number of folds for cross-validation (default = 5)')
    optionalCv.add_argument('-resume_fold', '--resume_from_fold', type = int, default=1, help='Resume cross-validation from the specified fold (default = 1)')
    optionalCv.add_argument('-tr_prop', '--train_prop', type = float, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalCv.add_argument('-bfactor', '--batch_factor', type = int, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalCv.add_argument('-loss', '--loss_function', choices=('weighted', 'nweighted'), default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalCv.add_argument('-gdir', '--gmdist_dir', help='Directory containing GM distance map images (default: --inp_dir).')
    optionalCv.add_argument('-vdir', '--ventdist_dir', help='Directory containing ventricle distance map images (default: --inp_dir).')
    optionalCv.add_argument('-nclass', '--num_classes', type = int, default=2, help='No. of classes to consider in the target labels; any additional class will be considered part of background (default=2)')
    optionalCv.add_argument('-plane', '--acq_plane', choices=('all', 'axial', 'sagittal', 'coronal'), default='all', help='The plane in which the model needs to be trained. Options: axial, sagittal, coronal, all (default=all)')
    optionalCv.add_argument('-da', '--data_augmentation', action='store_false', help='Applying data augmentation (default=True)')
    optionalCv.add_argument('-af', '--aug_factor', type = int, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalCv.add_argument('-sv_resume', '--save_resume_training', action='store_true', help='Whether to save and resume training in case of interruptions (default-False)')
    optionalCv.add_argument('-ilr', '--init_learng_rate', type = float, default=0.001, help='Initial LR to use in scheduler (default=0.001)')
    optionalCv.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, default=10, help='Milestones for LR scheduler (default=10)')
    optionalCv.add_argument('-gamma', '--lr_sch_gamma', type = float, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalCv.add_argument('-opt', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Optimizer used for training. Options: adam, sgd (default=adam)')
    optionalCv.add_argument('-eps', '--epsilon', type = float, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalCv.add_argument('-mom', '--momentum', type = float, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalCv.add_argument('-bs', '--batch_size', type = int, default=8, help='Batch size (default=8)')
    optionalCv.add_argument('-ep', '--num_epochs', type = int, default=60, help='Number of epochs (default=60)')
    optionalCv.add_argument('-es', '--early_stop_val', type = int, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalCv.add_argument('-int', '--intermediate', action='store_true', help='Saving intermediate prediction results for each subject (default=False)')
    optionalCv.add_argument('-sv', '--save_checkpoint', action='store_true', help='Whether to save any checkpoint (default=False)')
    optionalCv.add_argument('-sv_mod', '--save_full_model', action='store_true', help='If -sv=True, whether to save the whole model or just weights (default=False, i.e. to save just weights); if -sv=False, nothing will be saved')
    optionalCv.add_argument('-cp_type', '--cp_save_type', choices=('best', 'last', 'everyN'), default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalCv.add_argument('-cp_n', '--cp_everyn_N', type = int, default=10, help='If -cp_type=everyN, the N value')
    optionalCv.add_argument('-to', '--t1_only', action='store_true', help='Only use T1 images (ignore FLAIR images if present)')
    optionalCv.add_argument('-fo', '--flair_only', action='store_true', help='Only use FLAIR images (ignore T1 images if present)')
    optionalCv.add_argument('-v', '--verbose', action='store_true', help='Display debug messages (default=False)')

    args = parser.parse_args()

    if not op.isdir(args.inp_dir):
        raise ValueError(f'{args.inp_dir} does not appear to be a valid input directory')
    if hasattr(args, 'model_dir') and not op.isdir(args.model_dir):
        raise ValueError(f'{args.model_dir} does not appear to be a valid directory')
    if hasattr(args, 'label_dir') and not op.isdir(args.label_dir):
        raise ValueError(f'{args.label_dir} does not appear to be a valid directory')
    if hasattr(args, 'output_dir') and not op.isdir(args.output_dir):
        raise ValueError(f'{args.output_dir} does not appear to be a valid directory')

    if getattr(args, 'loss_function', None) == 'weighted':
        if args.gmdist_dir is None:
            args.gmdist_dir = args.inp_dir
        if args.ventdist_dir is None:
            args.ventdist_dir = args.inp_dir
        if not op.isdir(args.gmdist_dir):
            raise ValueError(f'{args.gmdist_dir} does not appear to be a valid GM distance files directory')
        if not op.isdir(args.ventdist_dir):
            raise ValueError(f'{args.ventdist_dir} does not appear to be a valid ventricle distance files directory')

    if hasattr(args, 'init_learng_rate') and not (0 <= args.init_learng_rate <= 1):
        raise ValueError('Initial learning rate must be between 0 and 1')

    if hasattr(args, 'lr_sch_gamma') and not (0 <= args.lr_sch_gamma <= 1):
        raise ValueError('Learning rate reduction factor must be between 0 and 1')

    if hasattr(args, 'train_prop') and not (0 <= args.train_prop <= 1):
        raise ValueError('Training data proportion must be between 0 and 1')

    if hasattr(args, 'batch_size') and args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1')
    if hasattr(args, 'num_epochs') and args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1')
    if hasattr(args, 'batch_factor') and args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1')

    if hasattr(args, 'early_stop_val') and not (1 <= args.early_stop_val <= args.num_epochs):
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs')
    if hasattr(args, 'aug_factor') and args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1')

    if hasattr(args, 'cp_save_type') and args.cp_save_type == 'everyN':
        if not (1 <= args.cp_everyn_N <= args.num_epochs):
            raise ValueError(
                'N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs')

    if hasattr(args, 'cp_load_type') and (args.cp_load_type == 'specific'):
        args.cp_load_type = 'everyN'
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch when using -cp_type is "specific"!')

    if hasattr(args, 'num_classes') and args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    if hasattr(args, 'cv_fold') and args.cv_fold < 1:
        raise ValueError('Number of folds cannot be 0 or negative')

    if hasattr(args, 'resume_from_fold') and args.resume_from_fold < 1:
        raise ValueError('Fold to resume cannot be 0 or negative')

    if args.command == 'train':
        truenet_commands.train(args)
    elif args.command == 'evaluate':
        truenet_commands.evaluate(args)
    elif args.command == 'fine_tune':
        truenet_commands.fine_tune(args)
    elif args.command == 'cross_validate':
        truenet_commands.cross_validate(args)
    else:
        parser.parse_args(["--help"])
        sys.exit(0)

if __name__ == "__main__":
    main()
