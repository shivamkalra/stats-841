import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy.io import loadmat
import pandas as pd

if os.environ.get("ENVIRONMENT") == 'sharcnet':
    matplotlib.use('Agg')

if os.environ.get('ENVIRONMENT') == 'sharcnet':
    print("Setting sharcnet environment")
    is_sharcnet = True
    dataset_directory = '/work/s6kalra/Shared/datasets/melbourne-university-seizure-prediction'
    output_directory = '/work/s6kalra/projects/stats-841/output'
else:
    is_sharcnet = False
    dataset_directory = '/home/skalra/mnt/sharcnet.work/Shared/datasets/melbourne-university-seizure-prediction'
    output_directory = '/home/skalra/mnt/sharcnet.work/projects/stats-841/output'

_safe_label_map = None


def create_dir_if_not_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print "Created: ", directory_path


def stitch_spectrograms_images(spectrograms, stitch_shape=(4, 4)):

    stitched_spectrogram = np.zeros(spectrograms[0].shape *
                                    np.array(stitch_shape))

    for i, sp in enumerate(spectrograms):
        # convert to stitched image indexes
        indx = np.array(np.unravel_index(i, stitch_shape)) * sp.shape
        stitched_spectrogram[indx[0]:indx[0] + sp.shape[0], indx[1]:indx[1] +
                             sp.shape[1]] = sp

    return stitched_spectrogram


def load_data(patient_id=0, is_test=False):

    mat_files = get_mat_files(patient_id, is_test)
    labels = np.asarray(
        [get_labels(os.path.basename(mat_file)) for mat_file in mat_files])

    segments = labels[:, 1]

    if not is_test:
        targets = labels[:, 2]
        return {
            'dtype': 'train',
            'mat_file': mat_files,
            'segment': segments,
            'target': targets,
            'patient_id': patient_id
        }

    return {
        'dtype': 'test',
        'mat_file': mat_files,
        'segment': segments,
        'patient_id': patient_id
    }


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
    # generate labels for test data
    fname = os.path.splitext(file_name)[0]
    if fname.startswith('new'):
        return np.array(list(map(int, fname.split('_')[1:])))

    return np.array(list(map(int, fname.split('_'))))


def get_mat_files(patient_id=0, is_test=False):
    """
    Get all the mat files sorted in specific order. It is important to note that each patient data
    is segregated into separate folders.

    Args:
     patient_id: Patient index (0, 1, 2)

    Returns:
     list: absolute paths of all mat files for given patient.
    """
    dir_format = 'train_{0}'
    if is_test:
        dir_format = 'test_{0}_new'
    patient_directory = os.path.join(dataset_directory,
                                     dir_format.format(patient_id + 1))
    return sorted(glob.glob(os.path.join(patient_directory, '*.mat')))


def read_mat_file(mat_file_path):
    """
    Reads the mat file and returns tuple containing EEG data its properties (labels and TDL)
    respectively.

    Returns:
      tuple(ndarray,dict): Tuple consisting of EEG data
      for each channel and properties for that data (labels and flags).
    """

    # some MAT files are corrupted see:
    # https://www.kaggle.com/c/melbourne-university-seizure-prediction/forums/t/23356/some-of-data-corrupted/134307
    patient_data = loadmat(
        mat_file_path, verify_compressed_data_integrity=False)
    channel_data = patient_data['dataStruct'][0][0][0]

    return channel_data, {'label': get_labels(os.path.basename(mat_file_path))}


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


def normalize_spectrogram(spec):

    divisor = np.max(np.abs(spec))

    # some data is corrupted for max(abs(data)) gives 0 as everything is zero
    if not divisor == 0.:
        return spec / divisor

    return spec


def convert_to_spectrogram(patient_data_channel, time_n=60, plot=False):
    """
     Convert EEG data for given channel into a spectrogram.

     Args:
      patient_data_channel: EEG data for single channel.
      shape: Shape of the spectrogram (time_axis, frequency_axis).
      plot: Flag suggesting if spectrogram should be plotted.

     Returns:
      tuple: Normalized [-1, 1], 6x10 spectrogram which can be saved as image
     """

    NFFT = (len(patient_data_channel) / time_n)
    spec, freqs, t, ax = pylab.specgram(
        patient_data_channel, NFFT=NFFT, Fs=400, noverlap=0)

    if not plot:
        plt.clf()

    res_spec = []
    std_spec = []

    # ranges are taken from
    # https://irakorshunova.github.io/2014/11/27/seizures.html
    # further search Ira Korshunova's thesis
    # delta, theta, alpha, beta, lowgamma, highgamma
    freq_window_ranges = [(0.1, 4), (4, 8), (8, 12), (12, 30), (30, 70), (70,
                                                                          180)]
    freq_window_indexes = [
        np.where(freqs >= freq_l) and np.where(freqs <= freq_h)
        for freq_l, freq_h in freq_window_ranges
    ]

    # TODO: Why do we need reversed? I used because plt.imshow shows inverted
    # (well that's obvious) but does it make any difference?
    for spec_col in spec.T:
        res_spec.append(
            [np.mean(spec_col[idx]) for idx in reversed(freq_window_indexes)])
        std_spec.append(
            [np.std(spec_col[idx]) for idx in reversed(freq_window_indexes)])

    res_spec = normalize_spectrogram(np.array(res_spec))
    std_spec = normalize_spectrogram(np.array(std_spec))

    return res_spec.T, std_spec.T


def get_data_output_path_for_patient(patient_id,
                                     create_if_not_exists=False,
                                     dtype='train'):
    data_output_path = os.path.join(output_directory, "p{0}{1}".format(
        patient_id + 1, dtype))

    if create_if_not_exists:
        create_dir_if_not_exist(data_output_path)

    return data_output_path


def load_data_for_patient(patient_id, dtype='train', file_name='data.npy'):

    data_file_path = os.path.join(
        get_data_output_path_for_patient(
            patient_id, dtype=dtype), file_name)
    print data_file_path
    data = np.load(data_file_path)

    if data.dtype == 'O':
        return data.item()

    return data


def save_data_for_patient(patient_id, data, dtype='train', file_name='data.npy'):

    data_file_path = os.path.join(
        get_data_output_path_for_patient(
            patient_id, dtype=dtype), file_name)

    np.save(data_file_path, np.array(data))
    return data


def get_safe_index(mat_file_names):
    return np.where(
        [_safe_label_map[file_name] for file_name in mat_file_names])


if _safe_label_map is None:
    _safe_label_map = {}
    safe_label_path = os.path.join(dataset_directory,
                                   'train_and_test_data_labels_safe.csv')
    df = pd.read_csv(safe_label_path)
    for image_name, safe in zip(df['image'], df['safe']):
        _safe_label_map[image_name] = (safe == 1)
