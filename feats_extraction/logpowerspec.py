import numpy as np
import librosa
from feats_extraction.generic import stft, cqt, load_wav, preemphasis, load_wav_snf

# def logpowspec(wav_path, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=None, a_min=1e-30):
#     """Compute log power magnitude spectra (logspec).
#     Returns:
#         D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]
#     """
#     wav = load_wav_snf(wav_path)
#     if pre_emphasis is not None:
#         wav = preemphasis(wav, k=pre_emphasis)
#     spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     mag_spec = np.abs(spec)
#     powspec = 1.0 / n_fft * np.square(mag_spec)
#     powspec[powspec <= a_min] = a_min
#     lps = 10 * np.log10(powspec)
#     return lps


def logmagspec(wav_path, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=None):
    """Compute log power magnitude spectra (logspec).
    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]
    """
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    mag_spec = np.abs(spec)
    mag_spec[mag_spec <= 1e-30] = 1e-30
    lms = 10 * np.log10(mag_spec)
    return lms


def logpowcqt(wav_path, sr=16000, hop_length=512, n_bins=528, bins_per_octave=48, window="hann", fmin=3.5, pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    """Compute log power magnitude spectra (logspec).

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.
    amin : float > 0 [scalar], ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
        minimum threshold for `abs(S)` and `ref`
    
    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``
    """
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    cqtfeats = cqt(wav, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, fmin=fmin)
    magcqt = np.abs(cqtfeats)
    powcqt = np.square(magcqt)
    logpowcqt = librosa.power_to_db(powcqt, ref, amin, top_db)
    return logpowcqt

def logpowspec(wav_path, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    """Compute log power magnitude spectra (logspec).

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.
    amin : float > 0 [scalar], ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
        minimum threshold for `abs(S)` and `ref`
    
    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``
    """
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    # spec = spec[:-2, :]  # TODO: check why there are two abnormal frames.
    mag_spec = np.abs(spec)
    powspec = np.square(mag_spec)
    logpowspec = librosa.power_to_db(powspec, ref, amin, top_db)
    return logpowspec

def logpowspec_multichannel(wav_path, channel, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    """Compute log power magnitude spectra (logspec).

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.
    amin : float > 0 [scalar], ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
        minimum threshold for `abs(S)` and `ref`
    
    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``
    """
    wav = load_wav_snf(wav_path)
    try:
        wav = wav[:, channel]
    except:
        wav = wav
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    # spec = spec[:-2, :]  # TODO: check why there are two abnormal frames.
    mag_spec = np.abs(spec)
    powspec = np.square(mag_spec)
    logpowspec = librosa.power_to_db(powspec, ref, amin, top_db)
    return logpowspec



if __name__ == '__main__':
    wav_path = '/apdcephfs/private_nenali/lixu/Data_Source/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0000001.flac'
    print(wav_path)
    lpcqt = logpowcqt(wav_path)
    print(lpcqt.shape)

