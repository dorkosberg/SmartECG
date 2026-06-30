import argparse


def parse_opts():
    parser = argparse.ArgumentParser(
        description='General binary ECG classifier (Normal vs Abnormal) trained on MIT-BIH patients.'
    )

    # |----------------------------------------- Init settings ------------------------------------------------------|
    parser.add_argument(
        '--data_path',
        default='./dataset/mitbih_database/',
        type=str,
        help='The data directory path under which the dataset lies.')
    parser.add_argument(
        '--output_path',
        default='./output/general_model/',
        type=str,
        help='Directory for checkpoints, logs, and patient split metadata.')
    parser.add_argument(
        '--mode',
        default='general-training',
        type=str,
        choices=['general-training', 'test'],
        help='general-training: train one model on multiple patients; test: evaluate a saved checkpoint.')
    parser.add_argument(
        '--model_path',
        default=None,
        type=str,
        help='Path to a saved checkpoint (.pth). Required for --mode test.')
    parser.add_argument(
        '--splits_path',
        default=None,
        type=str,
        help='Path to patient_splits.json. Defaults to the file next to model_path or output_path.')
    parser.add_argument(
        '--input_size',
        default=128,
        type=int,
        help='The size of each pulse-width window')
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='Random seed used for patient-level train/val/test splits.')
    parser.add_argument(
        '--val_ratio',
        default=0.15,
        type=float,
        help='Fraction of patients reserved for validation.')
    parser.add_argument(
        '--test_ratio',
        default=0.15,
        type=float,
        help='Fraction of patients reserved for held-out testing.')

    # |------------------------------------------ CNN default settings ----------------------------------------------|
    parser.add_argument(
        '--num_blocks',
        default=4,
        type=int,
        help='Number of blocks')
    parser.add_argument(
        '--block_channels',
        default=32,
        type=int,
        help='Block channels')
    parser.add_argument(
        '--kernel_size',
        default=5,
        type=int,
        help='The convolution kernel size of CNN')

    # |--------------------------------------- Training global settings ----------------------------------------------|
    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str,
        help='(Adam | SGD)')
    parser.add_argument(
        '--lr_scheduler',
        default='reducelr',
        type=str,
        help='(reducelr | cycliclr | cosAnnealing)')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight decay hyperparameter value of optimizer')
    parser.add_argument(
        '--n_epochs',
        default=30,
        type=int,
        help='The maximum number of total epochs to run.')

    # |---------------------------------------- General training settings --------------------------------------------|
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch size used during training.')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--weighted_sampling',
        type=bool,
        default=True,
        help='Enable weighted sampling during training.'
    )

    args = parser.parse_args()

    return args
