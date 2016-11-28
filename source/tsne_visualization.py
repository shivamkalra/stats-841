import utils as ut
import numpy as np
import seaborn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from os.path import basename
from sklearn.preprocessing import MinMaxScaler


def chilyfy_data(data):
    data['file_name'] = np.array([basename(fn) for fn in data['mat_files']])
    data['mat_files'] = np.array(data['mat_files'])
    data['data'] = np.array(data['data'])
    data['target'] = np.array(data['target'])
    return data


patient_id = 0

data2 = ut.apply_safe_indexes(
    chilyfy_data(
        ut.load_data_for_patient(
            patient_id, file_name='traditional.npy')))
freq_level = 1
X = data2['data']
scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)

# use this scaler for all the testing data transformation
# otherwise you're doomed because some Butter filter features
# are very large crazy numbers
model = TSNE(n_components=2, verbose=1)
Y = model.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], c=data2['target'])
plt.show()
