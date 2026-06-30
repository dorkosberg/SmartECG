import os
import random

import numpy as np
import torch

from opts import parse_opts
from running import run_test, run_training


def discover_patients(data_path):
    patients = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            patients.append(os.path.basename(os.path.join(data_path, file))[:3])
    return sorted(patients)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    opt = parse_opts()
    set_seed(opt.seed)

    patients = discover_patients(opt.data_path)
    if not patients:
        raise FileNotFoundError(f"No MIT-BIH annotation files found in {opt.data_path}")

    if opt.mode == "general-training":
        # One general model is trained on multiple patients together.
        # Per-patient fine-tuning / individual-model loops were removed.
        metrics = run_training(opt, patients)
        print("\nGeneral training complete.")
        print(f"Best model saved to: {os.path.join(opt.output_path, 'best_model.pth')}")
        print(f"Final held-out test accuracy: {metrics['accuracy']:.2f}%")
    elif opt.mode == "test":
        metrics = run_test(opt, patients)
        print("\nEvaluation complete.")
        print(f"Held-out test accuracy: {metrics['accuracy']:.2f}%")
    else:
        raise ValueError(f"Unsupported mode: {opt.mode}")


if __name__ == '__main__':
    main()
