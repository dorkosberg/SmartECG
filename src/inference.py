import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scipy.signal import resample
from data_processing import butter_bandpass_filter, resample_beats, z_norm
from models import cnn1D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 360
FS = 360
INPUT_SIZE = 128
MAX_WINDOWS = 300
PLOT_SECONDS = 15


ECG_COLUMN_HINTS = (
    'ecg', 'mlii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
    'lead', 'signal', 'value', 'amplitude', 'volt', 'beat', 'channel',
)
INDEX_COLUMN_HINTS = (
    'sample', 'index', 'time', 'id', '#', 'timestamp', 'sec', 'second', 'frame',
)


def _read_csv_flexible(file_obj, nrows=None):
    """Read CSV/TSV with auto delimiter detection and common encodings."""
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    raw = file_obj.read()
    if isinstance(raw, bytes):
        for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode('utf-8', errors='replace')
    else:
        text = raw

    errors = []
    for sep in (None, ',', ';', '\t', '|'):
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine='python',
                nrows=nrows,
            )
            if df.shape[1] >= 1 and df.shape[0] >= 1:
                return df
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError(
        'לא ניתן לקרוא את הקובץ. ודא שזה CSV עם עמודת ערכים מספרית של אות ECG.'
    )


def _is_index_like(series, column_name):
    name = str(column_name).lower().strip().strip("'\"")
    if any(hint in name for hint in INDEX_COLUMN_HINTS):
        return True

    numeric = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric) < 20:
        return False

    diffs = numeric.diff().dropna()
    if len(diffs) == 0:
        return False

    # Sample counter: 0,1,2,... or 0.0,1.0,2.0,...
    if (diffs.round(6).nunique() == 1) and (abs(diffs.mean() - 1.0) < 0.01):
        return True

  # Time column in seconds: monotonic increasing, low variance relative to signal
    if 'time' in name and numeric.is_monotonic_increasing:
        return True

    return False


def _pick_ecg_column(df):
    """Pick the most likely ECG amplitude column from arbitrary CSV layouts."""
    scored = []

    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        valid_ratio = series.notna().mean()
        if valid_ratio < 0.5:
            continue

        series = series.dropna().reset_index(drop=True)
        if len(series) < 50:
            continue

        if _is_index_like(series, col):
            continue

        name = str(col).lower()
        name_bonus = 2.0 if any(hint in name for hint in ECG_COLUMN_HINTS) else 0.0
        variance = float(series.var()) if series.var() > 0 else 0.0
        scored.append((name_bonus + variance, col, series))

    if not scored:
        raise ValueError(
            'לא נמצאה עמודת ECG תקינה. הקובץ צריך לכלול לפחות עמודה אחת עם ערכים מספריים של אות ECG.'
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    _, column_name, signal = scored[0]
    return signal, str(column_name)


def resample_signal_to_fs(signal, source_fs, target_fs=FS):
    """Resample a full ECG trace to the model sampling rate (360 Hz)."""
    source_fs = float(source_fs)
    if abs(source_fs - target_fs) < 1e-6:
        return signal.reset_index(drop=True)

    new_length = max(50, int(round(len(signal) * target_fs / source_fs)))
    resampled = resample(signal.values, new_length)
    return pd.Series(resampled, name='beat')


def load_signal_from_csv(file_obj, sampling_rate=FS):
    """Load ECG signal from CSV files in many common layouts."""
    max_samples = WINDOW_SIZE * MAX_WINDOWS + WINDOW_SIZE
    if sampling_rate != FS:
        max_samples = int(max_samples * sampling_rate / FS) + WINDOW_SIZE

    df = _read_csv_flexible(file_obj, nrows=max_samples)
    signal, column_name = _pick_ecg_column(df)
    return signal, column_name


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


def run_inference(file_obj, model_path, sampling_rate=FS):
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    raw_signal, ecg_column = load_signal_from_csv(file_obj, sampling_rate=sampling_rate)
    raw_signal = resample_signal_to_fs(raw_signal, sampling_rate, FS)
    processed = preprocess_signal(raw_signal)
    windows = segment_to_windows(processed)
    inputs, num_windows = prepare_model_inputs(windows)
    model = get_model(model_path)
    probabilities = predict_windows(model, inputs)
    summary = aggregate_predictions(probabilities)
    plot_b64 = plot_ecg(processed, probabilities)

    duration_sec = round(len(raw_signal) / FS, 2)
    summary['duration_seconds'] = duration_sec
    summary['analyzed_seconds'] = duration_sec
    summary['ecg_column'] = ecg_column
    summary['sampling_rate'] = float(sampling_rate)
    summary['plot_base64'] = plot_b64
    return summary
