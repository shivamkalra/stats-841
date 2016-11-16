from scipy.io import loadmat
import glob
import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

if os.environ.get('ENVIRONMENT') is 'sharcnet':
    is_sharcnet = True
    dataset_directory = '/work/s6kalra/Shared/datasets/melbourne-university-seizure-prediction'
    output_directory = '/work/s6kalra/projects/stats-841/output'
else:
    is_sharcnet = False
    dataset_directory = '/home/skalra/mnt/sharcnet.work/Shared/datasets/melbourne-university-seizure-prediction'
    output_directory = '/home/skalra/projects/stats-841/outputs'


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

    return np.array(map(int, os.path.splitext(file_name)[0].split('_')))


def get_mat_files(patient_id=0):
    """
    Get all the mat files sorted in specific order. It is important to note that each patient data
    is segregated into separate folders.

    Args:
     patient_id: Patient index (0, 1, 2)

    Returns:
     list: absolute paths of all mat files for given patient.
    """
    patient_directory = os.path.join(dataset_directory,
                                     'train_{0}'.format(patient_id + 1))
    return sorted(glob.glob(os.path.join(patient_directory, '*.mat')))


def read_mat_file(mat_file_path):
    """
    Reads the mat file and returns tuple containing EEG data its properties (labels and TDL)
    respectively.

    Returns:
      tuple(ndarray,dict): Tuple consisting of EEG data
      for each channel and properties for that data (labels and flags).
    """
    patient_data = loadmat(mat_file_path)
    channel_data = patient_data['dataStruct'][0][0][0]

    return (channel_data, {
        'label': get_labels(os.path.basename(mat_file_path))
    })


def get_data_generator(patient_id=0):
    """
     Utility function that provides the generator to iterate for all the data for patient at index
    `patient_id`.

     Args:
      patient_id (int): Index of the patient. 0 means patient 1.

     Returns:
      generator [(ndarray,dict)]: generator of a tuple consisting of EEG data
      for each channel and properties for that data (labels and flags).
     """

    mat_files = get_mat_files(patient_id)

    for mat_file in mat_files:
        yield read_mat_file(mat_file)


def convert_to_spectrogram(patient_data_channel, plot=False):
    """
     Convert EEG data for given channel into a spectrogram.

     Args:
      patient_data_channel: EEG data for single channel.
      shape: Shape of the spectrogram (time_axis, frequency_axis).
      plot: Flag suggesting if spectrogram should be plotted.

     Returns:
      ndarray: Normalized spectrogram which can be saved as image
     """

    NFFT = (len(patient_data_channel) / 10)
    spec, freqs, t, ax = pylab.specgram(
        patient_data_channel, NFFT=NFFT, Fs=400, noverlap=0)

    if not plot:
        plt.clf()

    res_spec = []

    # ranges are taken from
    # https://irakorshunova.github.io/2014/11/27/seizures.html
    # further search Ira Korshunova's thesis
    # delta, theta, alpha, beta, lowgamma, highgamma
    freq_window_ranges = [(0.1, 4), (4, 8), (12, 30), (30, 70), (70, 180)]
    freq_window_indexes = [
        np.where(freqs >= freq_l) and np.where(freqs <= freq_h)
        for freq_l, freq_h in freq_window_ranges
    ]

    # TODO: Why do we need reversed? I used because plt.imshow shows inverted
    # (well that's obvious) but does it make any difference?
    for spec_col in spec.T:
        res_spec.append(
            [np.mean(spec_col[idx]) for idx in reversed(freq_window_indexes)])

    res_spec = np.array(res_spec)

    # applying pre-processing to normalize the flattened values between [0, 1]
    # copy=False means act on same array, thus I could return unraveled
    # res_spec back
    preprocessing.minmax_scale(res_spec.ravel(), copy=False)
    return res_spec.T


plt.set_cmap('Oranges')
it = get_data_generator(0)
it.next()
it.next()
it.next()
it.next()
it.next()
(data, prop) = it.next()
spec = convert_to_spectrogram(data[:, 0])

#plt.figure()
plt.imshow(spec, interpolation='none')
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()

# labels = np.array(map(get_labels, map(os.path.basename, mat_files)))

# mdata = loadmat(mat_files[83])
# wdata = mdata['dataStruct'][0][0][0]

# ch1 = wdata[:, 2]

# # generate specgram

# def create_bin(col):
#     return np.mean(col[:-1].reshape(3500, 6), axis=0)

# plt.set_cmap('hsv')
# Pxx, freqs, t, plot = pylab.specgram(
#     ch1, NFFT=2 * 21000, Fs=400, noverlap=int(21000))

# Pcc = [create_bin(Pxx[:, i]) for i in range(10)]
# Pcc = np.asarray(Pcc).T

# plt.figure()
