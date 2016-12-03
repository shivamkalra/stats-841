import utils as ut
import numpy as np
import seaborn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from os.path import basename
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def chilyfy_data(data):
    data['file_name'] = np.array([basename(fn) for fn in data['mat_files']])
    data['mat_files'] = np.array(data['mat_files'])
    data['data'] = np.array(data['data'])
    if 'target' in data:
        data['target'] = np.array(data['target'])
    return data


patient_id = 2

data = ut.apply_safe_indexes(
    chilyfy_data(
        ut.load_data_for_patient(
            patient_id, file_name='traditional.npy')))
X = data['data']
scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)

clf = SVC(verbose=True, probability=True, C=1e2)
clf.fit(X, data['target'])


y_pred = clf.predict(X)

print accuracy_score(data['target'], y_pred)


indexes = np.arange(data['target'].shape[0]).reshape(-1, 30)

res = [np.max(y_pred[indx]) for indx in indexes]

ac_res = [np.mean(data['target'][indx]) for indx in indexes]

data2 = chilyfy_data(
        ut.load_data_for_patient(
            patient_id, dtype='test', file_name='traditional.npy'))
X_ts = data2['data']

X_ts = scaler.transform(X_ts)

y_ts_pred = clf.predict(X_ts)

indexes_ts = np.arange(data2['mat_files'].shape[0]).reshape(-1, 30)

res_ts = [np.max(y_ts_pred[indx]) for indx in indexes_ts]

