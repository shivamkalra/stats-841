import os
import utils
import numpy as np
from os.path import basename
from scoop import futures


def create_dir_if_not_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print "Created: ", directory_path


def create_spectrogram_images(mat_file_path):

    data, properties = utils.read_mat_file(mat_file_path)

    spectrograms = []
    for data_channel in data.T:
        spectrograms.append(utils.convert_to_spectrogram(data_channel))

    return np.array(spectrograms)


def stitch_spectrograms_images(spectrograms, stitch_shape=(4, 4)):

    stitched_spectrogram = np.zeros(spectrograms[0].shape *
                                    np.array(stitch_shape))

    for i, sp in enumerate(spectrograms):
        # convert to stitched image indexes
        indx = np.array(np.unravel_index(i, stitch_shape)) * sp.shape
        stitched_spectrogram[indx[0]:indx[0] + sp.shape[0], indx[1]:indx[1] +
                             sp.shape[1]] = sp

    return stitched_spectrogram


def write_patient_data(pdata):

    output_dir = os.path.join(
        utils.output_directory,
        'p{0}{1}'.format(pdata['patient_id'] + 1, pdata['dtype']))

    create_dir_if_not_exist(output_dir)
    raw_spectrograms = []
    stitched_spectrograms = []
    for mat_file in pdata['mat_file']:
        sps = create_spectrogram_images(mat_file)
        raw_spectrograms.append(sps)
        stitched_spectrograms.append(stitch_spectrograms_images(sps))

    mat_file_names = map(basename, pdata['mat_file'])
    input_data = {
        'raw_spectrograms': np.array(raw_spectrograms),
        'file_name': np.array(mat_file_names),
        'segment': pdata['segment']
    }

    if pdata['dtype'] == 'train':
        input_data['target'] = pdata['target']

    np.save(os.path.join(output_dir, 'data.npy'), input_data)
    np.save(
        os.path.join(output_dir, 'spectrograms_4x4.npy'),
        np.array(stitched_spectrograms))


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
