#!/usr/bin/env python

import argparse
import sys
from truenet.true_net import (truenet_commands, truenet_help_messages)

#=========================================================================================
# FSL TRUE_NET
# Vaanathi Sundaresan
# 01-04-2021, Oxford
#=========================================================================================
def main():
    desc_msgs = truenet_help_messages.desc_descs()
    epilog_msgs = truenet_help_messages.epilog_descs()
    parser = argparse.ArgumentParser(prog='truenet', formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description=desc_msgs['mainparser'], epilog=epilog_msgs['mainparser'])
    subparsers = parser.add_subparsers(dest="command")

    parser_train = subparsers.add_parser('train', formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description=desc_msgs['train'], epilog=epilog_msgs['subparsers'])
    requiredNamedtrain = parser_train.add_argument_group('Required named arguments')
    requiredNamedtrain.add_argument('-i', '--inp_dir', type=str, required=True, help='Input directory containing training images')
    requiredNamedtrain.add_argument('-l', '--label_dir', type=str, required=True, help='Directory containing lesion manual masks')
    requiredNamedtrain.add_argument('-m', '--model_dir', type=str, required=True, help='Directory for saving model weights')
    optionalNamedtrain = parser_train.add_argument_group('Optional named arguments')
    optionalNamedtrain.add_argument('-tr_prop', '--train_prop', type = float, required=False, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalNamedtrain.add_argument('-bfactor', '--batch_factor', type = int, required=False, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalNamedtrain.add_argument('-loss', '--loss_function', type = str, required=False, default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalNamedtrain.add_argument('-gdir', '--gmdist_dir', type = str, required=False, default=None, help='Directory containing GM distance map images. Required if -loss=weighted')
    optionalNamedtrain.add_argument('-vdir', '--ventdist_dir', type = str, required=False, default=None, help='Directory containing ventricle distance map images. Required if -loss=weighted')
    optionalNamedtrain.add_argument('-nclass', '--num_classes', type = int, required=False, default=2,
                                help='No of classes to consider in the target labels; any additional class will be considered part of background (default=2)')
    optionalNamedtrain.add_argument('-plane', '--acq_plane', type = str, required=False, default='all', help='Options: axial, sagittal, coronal, all (default = all)')
    optionalNamedtrain.add_argument('-da', '--data_augmentation', type = bool, required=False, default=True, help='Applying data augmentation (default=True)')
    optionalNamedtrain.add_argument('-af', '--aug_factor', type = int, required=False, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalNamedtrain.add_argument('-sv_resume', '--save_resume_training', type = bool, required=False, default=False, help='Whether to save and resume training in case of interruptions (default-False)')
    optionalNamedtrain.add_argument('-ilr', '--init_learng_rate', type = float, required=False, default=0.001, help='Initial LR to use in scheduler (default=0.001)')
    optionalNamedtrain.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, required=False, default=10, help='Milestones for LR scheduler (default=10)')
    optionalNamedtrain.add_argument('-gamma', '--lr_sch_gamma', type = float, required=False, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalNamedtrain.add_argument('-opt', '--optimizer', type = str, required=False, default='adam', help='Optimizer used for training. Options: adam, sgd (default=adam)')
    optionalNamedtrain.add_argument('-eps', '--epsilon', type = float, required=False, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalNamedtrain.add_argument('-mom', '--momentum', type = float, required=False, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalNamedtrain.add_argument('-bs', '--batch_size', type = int, required=False, default=8, help='Batch size (default=8)')
    optionalNamedtrain.add_argument('-ep', '--num_epochs', type = int, required=False, default=60, help='Number of epochs (default=60)')
    optionalNamedtrain.add_argument('-es', '--early_stop_val', type = int, required=False, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalNamedtrain.add_argument('-sv_mod', '--save_full_model', type = bool, required=False, default=False, help='Saving the whole model instead of weights alone (default=False)')
    optionalNamedtrain.add_argument('-cp_type', '--cp_save_type', type = str, required=False, default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalNamedtrain.add_argument('-cp_n', '--cp_everyn_N', type = int, required=False, default=10, help='If -cp_type=everyN, the N value (default=10)')
    optionalNamedtrain.add_argument('-v', '--verbose', type = bool, required=False, default=False, help='Display debug messages (default=False)')
    parser_train.set_defaults(func=truenet_commands.train)


    parser_evaluate = subparsers.add_parser('evaluate', formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description=desc_msgs['evaluate'], epilog=epilog_msgs['subparsers'])
    requiredNamedevaluate = parser_evaluate.add_argument_group('Required named arguments')
    requiredNamedevaluate.add_argument('-i', '--inp_dir', type=str, required=True, help='Input directory containing test images')
    requiredNamedevaluate.add_argument('-m', '--model_name', type=str, required=True, help='Model basename with absolute path'
                                                                                           'if you want to use pretrained model, use mwsc/ukbb')
    requiredNamedevaluate.add_argument('-o', '--output_dir', type=str, required=True, help='Directory for saving predictions')
    optionalNamedevaluate = parser_evaluate.add_argument_group('Optional named arguments')
    optionalNamedevaluate.add_argument('-cpu', '--use_cpu', action='store_true',
                                    help='Force the model to evaluate the model on CPU True/False (default=False)')
    # optionalNamedevaluate.add_argument('-p', '--pretrained_model', type=bool, required=False, default=False, help='Whether to use a standard pre-trained model (default=False)')
    # optionalNamedevaluate.add_argument('-pmodel', '--pretrained_model_name', type=str, required=False, default='mwsc', help='Pre-trained model to be used: mwsc, ukbb (default = mwsc)')
    optionalNamedevaluate.add_argument('-nclass', '--num_classes', type = int, required=False, default=2,
                                help='Number of classes in the labels used for training the model (for both pretrained models, -nclass=2) (default=2)')
    optionalNamedevaluate.add_argument('-int', '--intermediate', type=bool, required=False, default=False, help='Saving intermediate predictionss (individual planes) for each subject (default=False)')
    optionalNamedevaluate.add_argument('-cp_type', '--cp_load_type', type = str, required=False, default='last', help='Checkpoint to be loaded. Options: best, last, specific (default = last)')
    optionalNamedevaluate.add_argument('-cp_n', '--cp_everyn_N', type = int, required=False, default=None, help='If -cp_type=specific, the N value (default=10)')
    optionalNamedevaluate.add_argument('-v', '--verbose', type = bool, required=False, default=False, help='Display debug messages (default=False)')
    parser_evaluate.set_defaults(func=truenet_commands.evaluate)


    parser_finetune = subparsers.add_parser('fine_tune', formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description=desc_msgs['fine_tune'], epilog=epilog_msgs['subparsers'])
    requiredNamedft = parser_finetune.add_argument_group('Required named arguments')
    requiredNamedft.add_argument('-i', '--inp_dir', type=str, required=True, help='Input directory containing training images')
    requiredNamedft.add_argument('-l', '--label_dir', type=str, required=True, help='Directory containing lesion manual masks')
    requiredNamedft.add_argument('-m', '--model_name', type=str, required=True, help='Model basename with absolute path (will not be considered if optional argument -p=True)')
    requiredNamedft.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory for saving fine-tuned models/weights')
    optionalNamedft = parser_finetune.add_argument_group('Optional named arguments')
    optionalNamedft.add_argument('-p', '--pretrained_model', type=bool, required=False, default=False, help='Whether to use a standard pre-trained model (default=False)')
    optionalNamedft.add_argument('-pmodel', '--pretrained_model_name', type=str, required=False, default='mwsc', help='Pre-trained model to be used: mwsc, ukbb (default=mwsc')
    optionalNamedft.add_argument('-cpld_type', '--cp_load_type', type = str, required=False, default='last', help='Checkpoint to be loaded. Options: best, last, specific (default=last')
    optionalNamedft.add_argument('-cpld_n', '--cpload_everyn_N', type = int, required=False, default=10, help='If -cpld_type=specific, the N value (default=10)')
    optionalNamedft.add_argument('-ftlayers', '--ft_layers', nargs='+', type=int, required=False, default=2, help='Layers to fine-tune starting from the decoder (default=1 2)')
    optionalNamedft.add_argument('-tr_prop', '--train_prop', type = float, required=False, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalNamedft.add_argument('-bfactor', '--batch_factor', type = int, required=False, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalNamedft.add_argument('-loss', '--loss_function', type = str, required=False, default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalNamedft.add_argument('-gdir', '--gmdist_dir', type = str, required=False, default=None, help='Directory containing GM distance map images. Required if -loss=weighted')
    optionalNamedft.add_argument('-vdir', '--ventdist_dir', type = str, required=False, default=None, help='Directory containing ventricle distance map images. Required if -loss=weighted')
    optionalNamedft.add_argument('-plane', '--acq_plane', type = str, required=False, default='all', help='The plane in which the model needs to be fine-tuned. Options: axial, sagittal, coronal, all (default=all)')
    optionalNamedft.add_argument('-nclass', '--num_classes', type = int, required=False, default=2,
                                help='No. of classes to consider in the target labels; any additional class will be considered part of background (default=2)')
    optionalNamedft.add_argument('-da', '--data_augmentation', type = bool, required=False, default=True, help='Applying data augmentation (default=True)')
    optionalNamedft.add_argument('-af', '--aug_factor', type = int, required=False, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalNamedft.add_argument('-sv_resume', '--save_resume_training', type = bool, required=False, default=False, help='Whether to save and resume training in case of interruptions (default-False)')
    optionalNamedft.add_argument('-ilr', '--init_learng_rate', type = float, required=False, default=0.0001, help='Initial LR to use for fine-tuning in scheduler (default=0.0001)')
    optionalNamedft.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, required=False, default=10, help='Milestones for LR scheduler (default=10)')
    optionalNamedft.add_argument('-gamma', '--lr_sch_gamma', type = float, required=False, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalNamedft.add_argument('-opt', '--optimizer', type = str, required=False, default='adam', help='Optimizer used for training. Options:adam, sgd (default=adam)')
    optionalNamedft.add_argument('-eps', '--epsilon', type = float, required=False, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalNamedft.add_argument('-mom', '--momentum', type = float, required=False, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalNamedft.add_argument('-bs', '--batch_size', type = int, required=False, default=8, help='Batch size (default=8)')
    optionalNamedft.add_argument('-ep', '--num_epochs', type = int, required=False, default=60, help='Number of epochs (default=60)')
    optionalNamedft.add_argument('-es', '--early_stop_val', type = int, required=False, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalNamedft.add_argument('-sv_mod', '--save_full_model', type = bool, required=False, default=False, help='Saving the whole model instead of weights alone (default=False)')
    optionalNamedft.add_argument('-cp_type', '--cp_save_type', type = str, required=False, default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalNamedft.add_argument('-cp_n', '--cp_everyn_N', type = int, required=False, default=10, help='If -cp_type=everyN, the N value')
    optionalNamedft.add_argument('-v', '--verbose', type = bool, required=False, default=False, help='Display debug messages (default=False)')
    optionalNamedft.add_argument('-cpu', '--use_cpu', action='store_true',
                                    help='Perform model fine-tuning on CPU True/False (default=False)')
    parser_finetune.set_defaults(func=truenet_commands.fine_tune)


    parser_cv = subparsers.add_parser('cross_validate', formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description=desc_msgs['cross_validate'], epilog=epilog_msgs['subparsers'])
    requiredNamedcv = parser_cv.add_argument_group('Required named arguments')
    requiredNamedcv.add_argument('-i', '--inp_dir', type=str, required=True, help='Input directory containing images')
    requiredNamedcv.add_argument('-l', '--label_dir', type=str, required=True, help='Directory containing lesion manual masks')
    requiredNamedcv.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory for saving predictions (and models)')
    optionalNamedcv = parser_cv.add_argument_group('Optional named arguments')
    optionalNamedcv.add_argument('-fold', '--cv_fold', type = int, required=False, default=5, help='Number of folds for cross-validation (default = 5)')
    optionalNamedcv.add_argument('-resume_fold', '--resume_from_fold', type = int, required=False, default=1, help='Resume cross-validation from the specified fold (default = 1)')
    optionalNamedcv.add_argument('-tr_prop', '--train_prop', type = float, required=False, default=0.8, help='Proportion of data used for training (default = 0.8)')
    optionalNamedcv.add_argument('-bfactor', '--batch_factor', type = int, required=False, default=10, help='No. of subjects considered for each mini-epoch (default = 10)')
    optionalNamedcv.add_argument('-loss', '--loss_function', type = str, required=False, default='weighted', help='Applying spatial weights to loss function. Options: weighted, nweighted (default=weighted)')
    optionalNamedcv.add_argument('-gdir', '--gmdist_dir', type = str, required=False, default=None, help='Directory containing GM distance map images. Required if -loss=weighted')
    optionalNamedcv.add_argument('-vdir', '--ventdist_dir', type = str, required=False, default=None, help='Directory containing ventricle distance map images. Required if -loss=weighted')
    optionalNamedcv.add_argument('-nclass', '--num_classes', type = int, required=False, default=2,
                                help='No. of classes to consider in the target labels; any additional class will be considered part of background (default=2)')
    optionalNamedcv.add_argument('-plane', '--acq_plane', type = str, required=False, default='all', help='The plane in which the model needs to be trained. Options: axial, sagittal, coronal, all (default=all)')
    optionalNamedcv.add_argument('-da', '--data_augmentation', type = bool, required=False, default=True, help='Applying data augmentation (default=True)')
    optionalNamedcv.add_argument('-af', '--aug_factor', type = int, required=False, default=2, help='Data inflation factor for augmentation (default=2)')
    optionalNamedcv.add_argument('-sv_resume', '--save_resume_training', type = bool, required=False, default=False, help='Whether to save and resume training in case of interruptions (default-False)')
    optionalNamedcv.add_argument('-ilr', '--init_learng_rate', type = float, required=False, default=0.001, help='Initial LR to use in scheduler (default=0.001)')
    optionalNamedcv.add_argument('-lrm', '--lr_sch_mlstone', nargs='+', type=int, required=False, default=10, help='Milestones for LR scheduler (default=10)')
    optionalNamedcv.add_argument('-gamma', '--lr_sch_gamma', type = float, required=False, default=0.1, help='LR reduction factor in the LR scheduler (default=0.1)')
    optionalNamedcv.add_argument('-opt', '--optimizer', type = str, required=False, default='adam', help='Optimizer used for training. Options: adam, sgd (default=adam)')
    optionalNamedcv.add_argument('-eps', '--epsilon', type = float, required=False, default=1e-4, help='Epsilon for adam optimiser (default=1e-4)')
    optionalNamedcv.add_argument('-mom', '--momentum', type = float, required=False, default=0.9, help='Momentum for sgd optimiser (default=0.9)')
    optionalNamedcv.add_argument('-bs', '--batch_size', type = int, required=False, default=8, help='Batch size (default=8)')
    optionalNamedcv.add_argument('-ep', '--num_epochs', type = int, required=False, default=60, help='Number of epochs (default=60)')
    optionalNamedcv.add_argument('-es', '--early_stop_val', type = int, required=False, default=20, help='No. of epochs to wait for progress (early stopping) (default=20)')
    optionalNamedcv.add_argument('-int', '--intermediate', type=bool, required=False, default=False, help='Saving intermediate prediction results for each subject (default=False)')
    optionalNamedcv.add_argument('-sv', '--save_checkpoint', type = bool, required=False, default=False, help='Whether to save any checkpoint (default=False)')
    optionalNamedcv.add_argument('-sv_mod', '--save_full_model', type = bool, required=False, default=False,
                                help='If -sv=True, whether to save the whole model or just weights (default=False, i.e. to save just weights); if -sv=False, nothing will be saved')
    optionalNamedcv.add_argument('-cp_type', '--cp_save_type', type = str, required=False, default='last', help='Checkpoint saving options: best, last, everyN (default=last)')
    optionalNamedcv.add_argument('-cp_n', '--cp_everyn_N', type = int, required=False, default=10, help='If -cp_type=everyN, the N value')
    optionalNamedcv.add_argument('-v', '--verbose', type = bool, required=False, default=False, help='Display debug messages (default=False)')
    parser_cv.set_defaults(func=truenet_commands.cross_validate)


    args = parser.parse_args()

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
