import utils as ut
from scipy.signal import butter, lfilter
import numpy as np
import os


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_butter_features(ch):
    """
    Channel windowed data to butter features
    We love butter!!
    """

    bf_feat = []
    for lf, hf in freq_bands:
        bf = butter_bandpass_filter(ch, lf, hf, fs)
        bf_feat.append(np.sum(bf * bf))

    return np.array(bf_feat)


def get_fft_features(ch):

    fd = np.absolute(np.fft.fft(ch))
    freqs = np.fft.fftfreq(ch.shape[-1], 1. / fs)
    rfreqs = freqs[:np.argmax(freqs) + 1]

    fft_feat = []
    for lf, hf in freq_bands:
        feat = np.sum(fd[np.where((rfreqs >= lf) & (rfreqs <= hf))])
        fft_feat.append(feat)

    return np.array(fft_feat)


def get_corr_features(ch):

    xcorr_feat = []
    for i in range(ch.shape[0] - 1):
        for j in range(i + 1, ch.shape[0]):

            xcorr_feat.append(np.correlate(ch[i], ch[j])[0])

    return np.array(xcorr_feat)


def get_features(mat_file):

    d, _ = ut.read_mat_file(mat_file)

    # 5 seconds = 600/5 = 120 windows
    window_size = d.shape[0] / 120

    for i in range(0, d.shape[0], window_size):
        wd = d[i:i + window_size, :].T

        bf_feats = np.array(map(get_butter_features, wd))
        ff_feats = np.array(map(get_fft_features, wd))
        xcorr_feats = get_corr_features(wd)

        yield np.append(np.append(bf_feats, ff_feats), xcorr_feats)


patient_id = 0
dtype = 'train'

if 'TEST' in os.environ:
    dtype = 'test'

if 'PID' in os.environ:
    patient_id = int(os.environ['PID'])

pd_tr = ut.load_data(patient_id=patient_id, is_test=(dtype == 'test'))


# quick hack no time!!
if 'target' not in pd_tr:
    pd_tr['target'] = [-1]*len(pd_tr['mat_file'])


fs = 400
freq_bands = [(0.1, 4), (4, 8), (8, 12), (12, 30), (30, 80), (80, 180)]

X = []
MF = []
T = []

mat_idx = 1
div = len(pd_tr['mat_file']) / 100.

for mat_file, target in zip(pd_tr['mat_file'], pd_tr['target']):

    for feat in get_features(mat_file):
        X.append(feat)
        MF.append(mat_file)
        T.append(target)

    print "Done mat file", mat_idx / div
    mat_idx += 1

final_data = {'data': np.array(X),
              'target': np.array(T)}

from os.path import basename
final_data['file_name'] = map(basename, MF)

ut.save_data_for_patient(
    patient_id, final_data,
    dtype=dtype,
    file_name='traditional.npy')
