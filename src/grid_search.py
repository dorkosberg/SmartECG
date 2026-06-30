"""Deprecated: grid search was used for per-patient hyperparameter tuning.

The project now trains one general binary ECG classifier with patient-level splits.
Use `python src/main.py --mode general-training` instead.
"""

import sys


def main():
    print(
        "grid_search.py is deprecated. "
        "Per-patient hyperparameter search was removed with the patient-specific pipeline.\n"
        "Train the general model with:\n"
        "  python src/main.py --mode general-training"
    )
    sys.exit(1)


if __name__ == '__main__':
    main()
