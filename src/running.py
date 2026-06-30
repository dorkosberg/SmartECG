import os
import shutil

import torch

from torch import nn, optim
from torch.optim import lr_scheduler

from data_processing import createData, get_dataloader, load_patient_splits, save_patient_splits
from training.test import test
from training.train_epoch import train_epoch
from training.val_epoch import val_epoch
from models_generation import generate_model
from utils import WriteLogger


def _resolve_splits_path(opt):
    if opt.splits_path:
        return opt.splits_path
    if opt.model_path:
        return os.path.join(os.path.dirname(opt.model_path), 'patient_splits.json')
    return os.path.join(opt.output_path, 'patient_splits.json')


def _setup_output_dir(opt):
    # Patient-specific per-patient output folders were removed.
    save_file_path = opt.output_path
    if os.path.exists(save_file_path):
        shutil.rmtree(save_file_path)
    os.makedirs(save_file_path)
    return save_file_path


def _build_optimizer(opt, model):
    if opt.optimizer.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
    if opt.optimizer.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum if hasattr(opt, 'momentum') else 0.9,
            dampening=opt.dampening if hasattr(opt, 'dampening') else 0,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov if hasattr(opt, 'nesterov') else False,
        )
    raise ValueError(f"Unsupported optimizer: {opt.optimizer}")


def _build_scheduler(opt, optimizer, train_dataloader):
    if opt.lr_scheduler == 'reducelr':
        return lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    if opt.lr_scheduler == 'cycliclr':
        cycle_momentum = opt.optimizer.lower() != 'adam'
        return lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.001,
            max_lr=0.3,
            step_size_up=20,
            step_size_down=1000,
            cycle_momentum=cycle_momentum,
        )
    if opt.lr_scheduler == 'cosAnnealing':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_dataloader), eta_min=0, last_epoch=-1
        )
    return None


def run_training(opt, patients):
    """Train one general model on multiple patients with patient-level splits."""
    save_file_path = _setup_output_dir(opt)
    splits_path = os.path.join(save_file_path, 'patient_splits.json')

    x_train, x_val, y_train, y_val, x_test, y_test, patient_splits = createData(opt, patients)
    save_patient_splits(patient_splits, splits_path)

    train_dataloader = get_dataloader(
        x_train, y_train, batch_size=opt.batch_size, drop_last=False, weightedSampling=opt.weighted_sampling
    )
    val_dataloader = get_dataloader(
        x_val, y_val, batch_size=opt.batch_size, drop_last=False, weightedSampling=False
    )
    test_dataloader = get_dataloader(
        x_test, y_test, batch_size=opt.batch_size, drop_last=False, weightedSampling=False
    )

    model, _ = generate_model(opt)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = _build_optimizer(opt, model)
    scheduler = _build_scheduler(opt, optimizer, train_dataloader)

    train_logger = WriteLogger(
        os.path.join(save_file_path, 'train.log'),
        ['epoch', 'loss', 'lr', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score'],
    )
    val_logger = WriteLogger(
        os.path.join(save_file_path, 'val.log'),
        ['epoch', 'loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score'],
    )
    test_logger = WriteLogger(
        os.path.join(save_file_path, 'test.log'),
        ['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score', 'confusion_matrix'],
    )

    best_val_loss = float('inf')
    best_model_path = os.path.join(save_file_path, 'best_model.pth')

    for epoch in range(opt.n_epochs):
        train_state = train_epoch(epoch, train_dataloader, model, criterion, optimizer, train_logger)
        scheduler, val_loss = val_epoch(epoch, val_dataloader, model, criterion, scheduler, val_logger)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(train_state, best_model_path)
            print(f"Saved new best model at epoch {epoch} with validation loss {val_loss:.6f}")

    print(f"\nLoading best model from {best_model_path} for held-out patient evaluation.")
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    metrics = test(test_dataloader, model, criterion, test_logger, title='Held-out test patients')
    return metrics


def run_test(opt, patients):
    """Evaluate a saved general model on held-out test patients only."""
    if not opt.model_path:
        raise ValueError("--model_path is required when --mode test")

    splits_path = _resolve_splits_path(opt)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"Patient split file not found at {splits_path}. "
            "Train with --mode general-training first or pass --splits_path."
        )

    patient_splits = load_patient_splits(splits_path)
    _, _, _, _, x_test, y_test, _ = createData(opt, patients, patient_splits=patient_splits)
    test_dataloader = get_dataloader(
        x_test, y_test, batch_size=opt.batch_size, drop_last=False, weightedSampling=False
    )

    model, _ = generate_model(opt)
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.BCELoss(reduction="mean")
    save_file_path = os.path.dirname(opt.model_path)
    test_logger = WriteLogger(
        os.path.join(save_file_path, 'test_eval.log'),
        ['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score', 'confusion_matrix'],
    )

    metrics = test(test_dataloader, model, criterion, test_logger, title='Held-out test patients')
    return metrics
