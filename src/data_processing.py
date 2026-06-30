import json
import os

import numpy as np
import pandas as pd
import torch

from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler


def z_norm(signal):
    """Normalizes the input signal of a single patient; fits all values between 0 and 1."""
    return (signal - min(signal)) / (max(signal) - min(signal))


def butter_bandpass_filter(sig, low_cut, high_cut, fs, order=3):
    b, a = butter(order, [low_cut, high_cut], fs=fs, btype='bandpass', output='ba')
    y = filtfilt(b, a, sig)
    return y


def get_patientData(dataPath, patient, norm=True, movingAverage=False, window=10, bandpassFilter=False, low_cut=0.4,
                    high_cut=30, fs=360):
    # Load current patient signal
    signal = pd.read_csv(os.path.join(dataPath, patient + ".csv"), usecols=[0, 1])
    # Load current patient signal annotations
    annotations = pd.read_fwf(os.path.join(dataPath, patient + "annotations.txt"), usecols=[0, 1, 2])

    signal.columns = ['sample_num', 'beat']
    annotations.columns = ['time', 'sample_num', 'type']

    if norm:
        # Normalize signal to the range [0, 1]
        signal.beat = z_norm(signal.beat)

    if movingAverage:
        # Apply moving average to the signal
        signal.beat = signal.beat.rolling(int(window)).mean()

    if bandpassFilter:
        # Pass the signal through a frequency bandpass filter
        signal.beat = butter_bandpass_filter(signal.beat, low_cut, high_cut, fs, order=3)

    return signal, annotations


def split_to_windows(beat, annotations, window_size=360):
    """Segment one patient's ECG into 1-second windows and map beat annotations to binary labels.

    Normal = 0, Abnormal = 1. Unclassified beats ('Q', '?', non-beat) are discarded.
    """

    # Create a dataframe where each row is one window. The last window is padded.
    windows = beat.reindex(range((beat.size // window_size + 1) * window_size), method='pad')
    windows = windows.values.reshape((-1, window_size))
    # Set normal labels
    normal_beats = ['N']
    # Set abnormality labels
    abnormal_beats = ['L', 'R', 'V', 'A', 'a', 'j', 'S', 'F', 'E', 'e', 'r', '!', 'f', '/']
    labels = [-1] * windows.size
    for index, row in annotations.iterrows():
        idx = row['sample_num']
        ann = row['type']
        if ann in normal_beats:
            labels[idx] = 0
        elif ann in abnormal_beats:
            labels[idx] = 1
    labels = np.asarray(labels)
    labels = labels.reshape((-1, window_size))
    w_labels = np.amax(labels, axis=1)
    # Find all indexes with labels <> -1
    kept_idx = np.asarray(w_labels != -1).nonzero()[0]
    # Keep relevant windows and labels (i.e., remove samples with label -1)
    k_windows = windows[kept_idx, :]
    k_labels = w_labels[kept_idx]

    return k_windows, k_labels


def resample_beats(beats, sample_size):
    """Resamples input beats to fixed dimension."""

    beats_resampled = np.zeros((len(beats), sample_size))
    for i in range(beats_resampled.shape[0] - 1):
        beats_resampled[i] = resample(beats[i], sample_size)

    return beats_resampled


def exclude_patients(labels):
    labels_distribution = np.array(np.unique(labels, return_counts=True)).T

    if labels_distribution.shape[0] == 1:
        return True
    elif labels_distribution[1][1] / (labels_distribution[0][1] + labels_distribution[1][1]) < 0.009:
        return True
    else:
        return False


def resize_input(data):
    # Convert (N, D) to (N, 1, D) to fit the 1D CNN
    return data.reshape((data.shape[0], 1, data.shape[1]))


def load_patient_windows(opt, patient):
    """Load, filter, normalize, segment, and resample one patient's ECG beats."""
    signal, signal_notes = get_patientData(
        opt.data_path, patient, norm=True, movingAverage=False, bandpassFilter=True
    )
    windows, labels = split_to_windows(signal.beat, signal_notes)
    windows_resampled = resample_beats(windows, opt.input_size)
    return windows_resampled, labels


def load_patients_data(opt, patient_ids):
    """Load preprocessed beat windows for each patient, excluding extreme class-imbalance cases."""
    patient_data = {}
    for patient in patient_ids:
        windows, labels = load_patient_windows(opt, patient)
        if exclude_patients(labels):
            print(f"Excluding patient {patient} due to class imbalance.")
            continue
        patient_data[patient] = (windows, labels)
    return patient_data


def split_patients_by_id(patient_ids, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split patients into train / validation / test sets.

    Beats from the same patient are never mixed across splits.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    train_val_patients, test_patients = train_test_split(
        patient_ids,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
    )
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=relative_val_ratio,
        random_state=random_state,
        shuffle=True,
    )

    return {
        'train': sorted(train_patients),
        'val': sorted(val_patients),
        'test': sorted(test_patients),
    }


def concatenate_patient_splits(patient_data, patient_ids):
    """Concatenate beat windows from a list of patients into global arrays."""
    if not patient_ids:
        raise ValueError("Patient split is empty; no data available for this partition.")

    data_arrays = []
    label_arrays = []
    for patient in patient_ids:
        windows, labels = patient_data[patient]
        data_arrays.append(windows)
        label_arrays.append(labels)

    data = np.concatenate(data_arrays)
    labels = np.concatenate(label_arrays)
    return resize_input(data), labels


def save_patient_splits(patient_splits, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as split_file:
        json.dump(patient_splits, split_file, indent=2)


def load_patient_splits(splits_path):
    with open(splits_path, 'r', encoding='utf-8') as split_file:
        return json.load(split_file)


def createData(opt, all_patients, patient_splits=None):
    """Create train / validation / test datasets using patient-level splits only."""
    patient_data = load_patients_data(opt, all_patients)
    valid_patients = sorted(patient_data.keys())

    if not valid_patients:
        raise ValueError("No valid patients found after preprocessing and exclusion filters.")

    if patient_splits is None:
        patient_splits = split_patients_by_id(
            valid_patients,
            val_ratio=opt.val_ratio,
            test_ratio=opt.test_ratio,
            random_state=opt.seed,
        )
    else:
        for split_name in ('train', 'val', 'test'):
            unknown_patients = set(patient_splits[split_name]) - set(valid_patients)
            if unknown_patients:
                raise ValueError(
                    f"Split '{split_name}' references unknown or excluded patients: {sorted(unknown_patients)}"
                )

    x_train, y_train = concatenate_patient_splits(patient_data, patient_splits['train'])
    x_val, y_val = concatenate_patient_splits(patient_data, patient_splits['val'])
    x_test, y_test = concatenate_patient_splits(patient_data, patient_splits['test'])

    print(
        "Patient split sizes:",
        f"train={len(patient_splits['train'])},",
        f"val={len(patient_splits['val'])},",
        f"test={len(patient_splits['test'])}",
    )
    print("Train patients:", patient_splits['train'])
    print("Validation patients:", patient_splits['val'])
    print("Test patients:", patient_splits['test'])

    return x_train, x_val, y_train, y_val, x_test, y_test, patient_splits


def get_balanced_sampler(annotations):

    labels_distribution = np.array(np.unique(annotations, return_counts=True)).T
    # Here we create a dictionary with the class as the key and the indices on the annotations (and beats) list (array)
    # as the value
    class_counts = {row[0]: row[1] for row in labels_distribution}
    # The method we will use to balance the dataset is oversampling
    sample_weights = [1 / class_counts[i] for i in annotations]
    # Weights: kind of probability of sample to be selected
    # Num_samples: size of resulting dataset
    # Replacement = True necessary for any oversampling done
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(annotations), replacement=True)
    return sampler


def get_dataloader(data, labels, batch_size=32, drop_last=False, weightedSampling=False):
    # Set weighted sampler
    sampler = None

    if weightedSampling:
        sampler = get_balanced_sampler(labels)

    # Convert np.array to Tensor
    data = torch.tensor(data).float()
    labels = torch.tensor(labels).float()
    dataset = list(zip(list(data), list(labels)))

    if sampler is None:
        sampler = RandomSampler(dataset)
    # Get dataloader
    dataloader = DataLoader(dataset=dataset,
                            num_workers=0,
                            batch_size=batch_size,
                            pin_memory=True,
                            sampler=sampler,
                            drop_last=drop_last)
    return dataloader
