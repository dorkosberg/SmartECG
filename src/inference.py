import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_processing import butter_bandpass_filter, resample_beats, z_norm
from models import cnn1D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 360
FS = 360
INPUT_SIZE = 128
MAX_WINDOWS = 300
PLOT_SECONDS = 15


def load_signal_from_csv(file_obj):
    """Load ECG signal from an uploaded CSV (MIT-BIH format: sample index + lead column)."""
    max_samples = WINDOW_SIZE * MAX_WINDOWS + WINDOW_SIZE
    df = pd.read_csv(file_obj, usecols=[0, 1], nrows=max_samples)
    if df.shape[1] < 2:
        raise ValueError('CSV must contain at least two columns: sample index and ECG values.')

    signal = df.iloc[:, 1].astype(float).reset_index(drop=True)
    signal.name = 'beat'
    return signal


def preprocess_signal(signal):
    """Apply the same filtering and normalization used during training."""
    if len(signal) < 50:
        raise ValueError('האות קצר מדי לניתוח. נדרשות לפחות ~50 דגימות ECG.')
    processed = signal.copy()
    if processed.max() == processed.min():
        processed = processed - processed.min() + 1e-6
    else:
        processed = z_norm(processed)
    processed = pd.Series(
        butter_bandpass_filter(processed.values, 0.4, 30, FS, order=3),
        name='beat',
    )
    return processed


def segment_to_windows(signal, window_size=WINDOW_SIZE):
    """Split signal into fixed-length windows (no annotations required)."""
    padded_len = (signal.size // window_size + 1) * window_size
    windows = signal.reindex(range(padded_len), method='pad')
    return windows.values.reshape((-1, window_size))


def prepare_model_inputs(windows, input_size=INPUT_SIZE):
    """Resample windows and reshape for the 1D CNN."""
    if len(windows) == 0:
        raise ValueError('Signal is too short to form ECG windows.')
    windows = windows[:MAX_WINDOWS]
    resampled = resample_beats(windows, input_size)
    tensor = torch.tensor(resampled, dtype=torch.float32).reshape(-1, 1, input_size)
    return tensor, len(windows)


_model = None
_model_path = None


def get_model(model_path):
    """Load the trained model once and reuse it for all requests."""
    global _model, _model_path
    if _model is None or _model_path != model_path:
        _model = load_trained_model(model_path)
        _model_path = model_path
    return _model


def load_trained_model(model_path, num_blocks=4, block_channels=32, kernel_size=5):
    model = cnn1D.cnn_1D(
        num_blocks=num_blocks,
        block_channels=block_channels,
        kernel_size=kernel_size,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_windows(model, inputs):
    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy().flatten()
    return outputs


def aggregate_predictions(probabilities):
    mean_prob = float(np.mean(probabilities))
    max_prob = float(np.max(probabilities))
    abnormal_windows = int(np.sum(probabilities >= 0.5))

    if mean_prob >= 0.5:
        label = 'Abnormal'
        label_he = 'לא תקין'
        confidence = mean_prob
    else:
        label = 'Normal'
        label_he = 'תקין'
        confidence = 1.0 - mean_prob

    return {
        'label': label,
        'label_he': label_he,
        'confidence_percent': round(confidence * 100, 2),
        'mean_abnormal_probability': round(mean_prob * 100, 2),
        'max_abnormal_probability': round(max_prob * 100, 2),
        'windows_analyzed': len(probabilities),
        'abnormal_windows': abnormal_windows,
        'normal_windows': len(probabilities) - abnormal_windows,
    }


def plot_ecg(signal, probabilities, window_size=WINDOW_SIZE):
    """Draw ECG trace with window-level predictions highlighted."""
    plot_samples = min(len(signal), PLOT_SECONDS * FS)
    x = np.arange(plot_samples) / FS
    y = signal.iloc[:plot_samples].values

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(x, y, color='#1a5276', linewidth=0.9, label='ECG (processed)')

    windows_in_plot = min(len(probabilities), plot_samples // window_size)
    for i in range(windows_in_plot):
        start = i * window_size
        end = min(start + window_size, plot_samples)
        if probabilities[i] >= 0.5:
            ax.axvspan(start / FS, end / FS, color='#e74c3c', alpha=0.18)

    ax.set_title('ECG Signal — red shading = windows classified as Abnormal', fontsize=11)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (normalized)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def run_inference(file_obj, model_path):
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    raw_signal = load_signal_from_csv(file_obj)
    processed = preprocess_signal(raw_signal)
    windows = segment_to_windows(processed)
    inputs, num_windows = prepare_model_inputs(windows)
    model = get_model(model_path)
    probabilities = predict_windows(model, inputs)
    summary = aggregate_predictions(probabilities)
    plot_b64 = plot_ecg(processed, probabilities)

    duration_sec = round(len(raw_signal) / FS, 2)
    summary['duration_seconds'] = duration_sec
    summary['plot_base64'] = plot_b64
    return summary
