from scipy.io import loadmat
import glob
import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

data_directory = '/home/skalra/mnt/sharcnet.work/Shared/datasets/melbourne-university-seizure-prediction'


def get_labels(file_name):
    """
    I_J_K.mat - the Jth training data segment corresponding to the Kth class
    (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three
    patients).

    Args:
    file_name (str): File name

    Returns:
    list: [I, J, K]
    """

    return map(int, os.path.splitext(file_name)[0].split('_'))


mat_files = glob.glob(os.path.join(data_directory, 'train_1/*.mat'))
labels = np.array(map(get_labels, map(os.path.basename, mat_files)))

mdata = loadmat(mat_files[83])
wdata = mdata['dataStruct'][0][0][0]

ch1 = wdata[:, 2]
# generate specgram


def create_bin(col):
    return np.mean(col[:-1].reshape(3500, 6), axis=0)

plt.set_cmap('hsv')
Pxx, freqs, t, plot = pylab.specgram(
    ch1,
    NFFT=2*21000,
    Fs=400,
    noverlap=int(21000))

Pcc = [create_bin(Pxx[:, i]) for i in range(10)]
Pcc = np.asarray(Pcc).T

plt.figure()

plt.imshow(scale(Pcc), interpolation='none')
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()


