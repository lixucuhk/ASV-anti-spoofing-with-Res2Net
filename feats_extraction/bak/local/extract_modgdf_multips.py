"""Module for extracting phase features
"""
import argparse

import numpy as np
from scipy.fftpack import dct
import scipy.io.wavfile as wavfile
import soundfile as sf
from python_speech_features.sigproc import preemphasis, framesig
import kaldi_io
from multiprocessing import Process
import os

# from data_reader.plot import plot_data


NFFT = 512
PREEMPH = 0.97
HAMMING_WINFUNC = np.hamming
LIFTER = 6
ALPHA = 0.4
GAMMA = 0.9


def get_complex_spec(wav_, winstep, winlen, with_time_scaled=False):
    """Return complex spec
    """
    # rate, sig = wavfile.read(wav_)
    sig, rate = sf.read(wav_, dtype=np.float32)

    sig = preemphasis(sig, PREEMPH)
    frames = framesig(sig, winlen * rate, winstep * rate, HAMMING_WINFUNC)
    complex_spec = np.fft.rfft(frames, NFFT)

    time_scaled_complex_spec = None
    if with_time_scaled:
        time_scaled_frames = np.arange(frames.shape[-1]) * frames
        time_scaled_complex_spec = np.fft.rfft(time_scaled_frames, NFFT)

    return complex_spec, time_scaled_complex_spec


def get_mag_spec(complex_spec):
    """Return mag spec
    """
    return np.absolute(complex_spec)


def get_phase_spec(complex_spec):
    """Return phase spec
    """
    return np.angle(complex_spec)


def get_real_spec(complex_spec):
    """Return real spec
    """
    return np.real(complex_spec)


def get_imag_spec(complex_spec):
    """Return imag spec
    """
    return np.imag(complex_spec)


def cepstrally_smoothing(spec):
    """Return cepstrally smoothed spec
    """
    _spec = np.where(spec == 0, np.finfo(float).eps, spec)
    log_spec = np.log(_spec)
    ceps = np.fft.irfft(log_spec, NFFT)
    win = (np.arange(ceps.shape[-1]) < LIFTER).astype(np.float)
    win[LIFTER] = 0.5
    return np.absolute(np.fft.rfft(ceps * win, NFFT))


def get_modgdf(complex_spec, complex_spec_time_scaled):
    """Get Modified Group-Delay Feature
    """
    mag_spec = get_mag_spec(complex_spec)
    cepstrally_smoothed_mag_spec = cepstrally_smoothing(mag_spec)
    #plot_data(cepstrally_smoothed_mag_spec, "cepstrally_smoothed_mag_spec.png", "cepstrally_smoothed_mag_spec")

    real_spec = get_real_spec(complex_spec)
    imag_spec = get_imag_spec(complex_spec)
    real_spec_time_scaled = get_real_spec(complex_spec_time_scaled)
    imag_spec_time_scaled = get_imag_spec(complex_spec_time_scaled)

    __divided = real_spec * real_spec_time_scaled \
            + imag_spec * imag_spec_time_scaled
    __tao = __divided / (cepstrally_smoothed_mag_spec ** (2. * GAMMA))
    __abs_tao = np.absolute(__tao)
    __sign = 2. * (__tao == __abs_tao).astype(np.float) - 1.
    return __sign * (__abs_tao ** ALPHA)

def get_modgdf_dct(complex_spec, complex_spec_time_scaled):
    """Get Modified Group-Delay Feature
    """
    mag_spec = get_mag_spec(complex_spec)
    cepstrally_smoothed_mag_spec = cepstrally_smoothing(mag_spec)
    # plot_data(cepstrally_smoothed_mag_spec, "cepstrally_smoothed_mag_spec.png", "cepstrally_smoothed_mag_spec")

    real_spec = get_real_spec(complex_spec)
    imag_spec = get_imag_spec(complex_spec)
    real_spec_time_scaled = get_real_spec(complex_spec_time_scaled)
    imag_spec_time_scaled = get_imag_spec(complex_spec_time_scaled)

    __divided = real_spec * real_spec_time_scaled \
            + imag_spec * imag_spec_time_scaled
    __tao = __divided / (cepstrally_smoothed_mag_spec ** (2. * GAMMA))
    __abs_tao = np.absolute(__tao)
    __sign = 2. * (__tao == __abs_tao).astype(np.float) - 1.
    return dct(__sign * (__abs_tao ** ALPHA), type=2, axis=1, norm='ortho')

def extract(wav_, winstep, winlen, mode): # mode = ['mgd', 'mgd_dct', 'mgd_dct_abs']

    complex_spec, complex_spec_time_scaled = get_complex_spec(wav_, winstep, winlen, with_time_scaled=True)
    if mode == 'mgd':
        return get_modgdf(complex_spec, complex_spec_time_scaled)
    elif mode == 'mgd_dct':
        return get_modgdf_dct(complex_spec, complex_spec_time_scaled)
    elif mode == 'mgd_dct_abs':
        return np.absolute(get_modgdf_dct(complex_spec, complex_spec_time_scaled))
    else: raise NameError('do not support mode: %s' %(mode))

def extract_file(wav_lines, wfilename, winstep, winlen, mode):

    ark_scp_output = 'ark:| copy-feats ark:- ark,scp:%s.ark,%s.scp' %(wfilename, wfilename)
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as wf:
        for line in wav_lines:
            items = line.split()
            key = items[0]
            wav_ = items[5]
            mat = extract(wav_, winstep, winlen, mode)
            kaldi_io.write_mat(wf, mat, key=key)


def main():
    """
    Main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", default="data/train/wav.scp")
    parser.add_argument("--wfilename", type=str)
    parser.add_argument('--nj', type=int, default=30)
    parser.add_argument("--winstep", type=float, default=0.01)
    parser.add_argument("--winlen", type=float, default=0.025)
    parser.add_argument("--mode", type=str, default='mgd')

    args = parser.parse_args()

    cdir = os.getcwd()
    wfilename = cdir+'/'+args.wfilename

    rfilename = args.wav_scp
    rfile = open(rfilename, 'r')
    wav_lines = rfile.readlines()
    rfile.close()

    wav_lines_split = np.array_split(wav_lines, args.nj)

    os.makedirs(os.path.dirname(wfilename), exist_ok=True)
    
    print('processing wav files with %s mode.' %(args.mode))

    processes = []
    for i, wav_batch in enumerate(wav_lines_split):
        print(f'Process {i} has been started.')
        wfile_batch = wfilename+'.%d' %(i)
        p = Process(target=extract_file, args=(wav_batch, wfile_batch, args.winstep, args.winlen, args.mode))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    

if __name__ == "__main__":
    main()
