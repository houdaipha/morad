import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import numpy as np
import librosa
import torch
import torch.nn.functional as F


# from openai whisper
def load_audio(file, sr):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length, *, axis=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels):
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio,
    n_mels=80,
    padding=0,
    device=None,
    n_fft=400,
    hop_length=160,
    sr=16000,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio, sr)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(
        audio,
        n_fft,
        hop_length,
        window=window,
        return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def spectrogram_to_image(log_spec, target_size, normalize=True):
    """
    Preprocess the log mel spectrogram to a 3-channel image

    Parameters:
    - log_spec (torch.Tensor): Log mel spectrogram
    - target_size (int): Target size

    Returns:
    - torch.Tensor: Preprocessed spectrogram of shape (3, target_size, target_size)
    """
    if not isinstance(log_spec, torch.Tensor):
        log_spec = torch.tensor(log_spec)

    if log_spec.dim() == 2:
        log_spec = log_spec.unsqueeze(0)

    resized_spec = F.interpolate(
        log_spec.unsqueeze(1),
        size=target_size,
        mode='bilinear',
        align_corners=False)

    if normalize:
        resized_spec = (resized_spec - resized_spec.min()) / (resized_spec.max() - resized_spec.min())

    output = resized_spec.repeat(1, 3, 1, 1)

    if output.size(0) == 1:
        output = output.squeeze(0)

    return output

def load_mfcc(
        path,
        target_length=None,
        sr=16000,
        n_mfcc=13,
        n_fft=400,
        hop_length=160,
        n_mels=128,
        deltas=False):
    y, audio_sr = librosa.load(path)
    if audio_sr != sr:
        y = librosa.resample(y, orig_sr=audio_sr, target_sr=sr)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if deltas:
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, delta1, delta2])
    if target_length is not None:
        mfcc = pad_or_trim(mfcc, target_length)
    return torch.from_numpy(mfcc.T)
