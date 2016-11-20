import utils as iu
import numpy as np

import csv
import os.path

from skimage.transform import radon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import svm

def apply_indexes(data, indexes):
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k][indexes]

    return new_data


def convert_to_image_format(raw_data):
    """
    Something better be not understood!!
    Trust me this work!
    """
    return np.rot90(np.fliplr(np.rot90(raw_data.T)), 2)


def write_submission(run_id):

    svms = [train_svm(i) for i in range(3)]
    results = [predict_for_patient(i, svms[i]) for i in range(3)]

    data = [['File', 'Class']]
    for res in results:
        for fn, c in res:
            data.append([fn, c])

    with open('results_r{0}.csv'.format(run_id), 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)


def predict_for_patient(patient_id, pred):

    print "predicting ", patient_id + 1
    data = iu.load_data_for_patient(patient_id, dtype='test')
    ch_idx = 4

    X = np.array([get_radon_features(sp[ch_idx]) for sp in data['raw_spectrograms']])

    return zip(data['file_name'], pred.predict(X))


def get_radon_features(im):
    thetas = np.linspace(0, 180, 4, False)
    rt = radon(im, sorted(thetas))
    # rt = np.array([rtt / np.max(rtt) for rtt in rt.T]).ravel()
    return rt.T.ravel()


def train_svm(patient_id):
    data = iu.load_data_for_patient(patient_id)
    safe_indexes = iu.get_safe_index(data['file_name'])

    data = apply_indexes(data, safe_indexes)
    ch_idx = 3

    X = np.array([get_radon_features(sp[ch_idx]) for sp in data['raw_spectrograms']])

    Y = map(int, data['target'])

    clf = svm.SVC()
    clf.fit(X, Y)

    return clf


write_submission(4)
