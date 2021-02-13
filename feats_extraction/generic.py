import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

def load_wav_snf(path):
    wav, sr = sf.read(path, dtype=np.float32)
    return wav

def load_wav(path, sr=16000):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr=16000):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # librosa.output.write_wav(path, wav.astype(np.int16), sr)
    wavfile.write(path, sr, wav.astype(np.int16))

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def inv_preemphasis(wav, k=0.97):
    return signal.lfilter([1], [1, -k], wav)


def cqt(wav, sr=16000, hop_length=512, n_bins=528, bins_per_octave=48, window='hann', fmin=3.5):
    tmp = librosa.cqt(y=wav, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, fmin=fmin) # [n_bins, t]
    return tmp.T # [t, n_bins]
    

def stft(wav, n_fft=512, hop_length=160, win_length=400, window="hann"):
    """Compute Short-time Fourier transform (STFT).
    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]
        STFT matrix
    """
    tmp = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)  # [1 + n_fft/2, t]
    return tmp.T  # (t, 1 + n_fft/2)

def inv_magphase(mag, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return mag * phase

def power_db_to_mag(power_db):
    power_spec = librosa.core.db_to_power(S_db=power_db, ref=1.0)
    mag_spec = np.sqrt(power_spec) 
    return mag_spec

# def revert_power_db_to_wav(flac_file, power_db, n_fft=1724, hop_length=130, win_length=1724, window="blackman"):
#     wav = load_wav_snf(flac_file)
#     spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     _, phase = librosa.magphase(spec)
#     mag = power_db_to_mag(power_db).T
#     complex_specgram = mag * phase
#     audio = librosa.istft(complex_specgram, hop_length=hop_length, win_length=win_length, window=window)
#     return audio

def revert_power_db_to_wav(gt_spec, adv_power_db, n_fft=1724, hop_length=130, win_length=1724, window="blackman"):
    # wav = load_wav_snf(flac_file)
    # spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    _, phase = librosa.magphase(gt_spec.T)
    phase = phase[:, :adv_power_db.shape[0]]
    mag = power_db_to_mag(adv_power_db).T
    complex_specgram = mag * phase
    audio = librosa.istft(complex_specgram, hop_length=hop_length, win_length=win_length, window=window)
    return audio






