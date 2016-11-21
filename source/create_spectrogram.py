import os
import utils
import numpy as np
from os.path import basename
from scoop import futures
from utils import create_dir_if_not_exist


def create_spectrogram_images(mat_file_path):
    data, properties = utils.read_mat_file(mat_file_path)

    spectrograms = []
    for data_channel in data.T:
        spectrograms.append(utils.convert_to_spectrogram(data_channel))

    return np.array(spectrograms)


def write_patient_data(pdata):

    output_dir = os.path.join(
        utils.output_directory,
        'p{0}{1}'.format(pdata['patient_id'] + 1, pdata['dtype']))

    create_dir_if_not_exist(output_dir)
    raw_data = list(futures.map(create_spectrogram_images, pdata['mat_file']))

    raw_spectrograms = []
    std_spectrograms = []

    for el in raw_data:
        raw_spectrograms.append(el[:, 0, :, :])
        std_spectrograms.append(el[:, 1, :, :])

    print np.array(raw_spectrograms).shape
    mat_file_names = map(basename, pdata['mat_file'])
    input_data = {
        'raw_spectrograms': np.array(raw_spectrograms),
        'std_spectrograms': np.array(std_spectrograms),
        'file_name': np.array(mat_file_names),
        'segment': pdata['segment']
    }

    if pdata['dtype'] == 'train':
        input_data['target'] = pdata['target']

    np.save(os.path.join(output_dir, 'data.npy'), input_data)


def collect_and_write_data(p_idx):
    pid = int(p_idx / 2)

    is_test = p_idx % 2 != 0

    # write training data
    pdata = utils.load_data(pid, is_test=is_test)

    write_patient_data(pdata)

    return 1


if __name__ == "__main__":
    create_dir_if_not_exist(utils.output_directory)
    result = futures.map(collect_and_write_data, range(6))
    print list(result)
