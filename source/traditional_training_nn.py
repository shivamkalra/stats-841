import os
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'

import utils as ut
import numpy as np
from os.path import basename
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import KFold
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
import csv
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import hamming

def chilyfy_data(data):
    data['file_name'] = np.array([basename(fn) for fn in data['mat_files']])
    data['mat_files'] = np.array(data['mat_files'])
    data['data'] = np.array(data['data'])
    if 'target' in data:
        data['target'] = np.array(data['target'])
    return data


def build_model():

    model = Sequential()

    model.add(Dense(684, init='he_normal', input_dim=312))
    model.add(PReLU())
    model.add(Dropout(0.3))

    model.add(Dense(584, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(384, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Dense(2, init='he_normal'))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model



def get_feat(spred):
    l = len(spred)
    ispred = zip(range(spred.shape[-1]), spred>=0.5)
    cons = [np.array(list(v))[:, 0] for _, v in groupby(ispred, lambda x: x[1])]
    const = filter(lambda x: (spred >= 0.5)[x[0]], cons)
    constv = [spred[idx] for idx in const]

    s = np.argsort(spred)

    f1 = np.mean(spred)
    f2 = np.std(spred)

    f3 = np.product(spred[s[-4:]])
    f4 = np.product(spred[s[-5:]])
    f5 = np.product(spred[s[-6:]])

    f6 = np.sum(map(lambda v: sum_by(2, v), filter(lambda x: x.shape[-1] >= 2, constv)))
    f7 = np.sum(map(lambda v: sum_by(3, v), filter(lambda x: x.shape[-1] >= 3, constv)))
    f8 = np.sum(map(lambda v: sum_by(4, v), filter(lambda x: x.shape[-1] >= 4, constv)))
    f9 = np.sum(map(lambda v: sum_by(5, v), filter(lambda x: x.shape[-1] >= 5, constv)))
    f10 = np.sum(map(lambda v: sum_by(6, v), filter(lambda x: x.shape[-1] >= 6, constv)))

    f11 = np.where(spred >= 0.5)[0].shape[-1]/l

    feat = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

    return feat


def chilly_build(pid):
    data = ut.load_data_for_patient(patient_id=pid, file_name='traditional.npy', dtype='test')
    data = chilyfy_data(data)

    X = scaler.transform(data['data'])

    idx = [map(int, np.array(list(v))[:, 0]) for _, v in groupby(zip(np.arange(0, data['target'].shape[-1]), data['mat_files']), key=lambda x: x[1])]
    files = [data['file_name'][ix][0] for ix in idx]

    pred = [model.predict_proba(X[ix, :])[:, 1] for ix in idx]
    pfeats = map(get_feat, pred)

    return np.array(pfeats), files




data = ut.load_data_for_patient(patient_id=2, file_name='traditional.npy')
data = ut.apply_safe_indexes(chilyfy_data(data))

# filter all complete dropouts
good_data_indx = np.where(np.mean(np.abs(data['data'][:, 192:]), axis=1) != 0)

data = ut.apply_indexes(data, good_data_indx)
X = data['data']

scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)
Y = np_utils.to_categorical(data['target'])

wdata = ut.load_data(patient_id=0)

model = build_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

weights = np.ones(data['target'].shape)
weights[np.where(data['target'] == 1)] = 60

model.fit(X, Y, validation_split=0.4, sample_weight=weights, nb_epoch=30, callbacks=[early_stopping])

print confusion_matrix(data['target'], model.predict_classes(X))
print accuracy_score(data['target'], model.predict_classes(X))
print roc_auc_score(data['target'], model.predict_proba(X)[:,1])

input()

from itertools import groupby, combinations

idx = [map(int, np.array(list(v))[:, 0]) for _, v in groupby(zip(np.arange(0, data['target'].shape[-1]), data['mat_files']), key=lambda x: x[1])]


def sum_by(n, l):
    _sum = 0
    for i in range(len(l) - n + 1):
        _sum += np.product(l[i:i+n])
    return _sum

turget = [np.mean(data['target'][ix]) for ix in idx]
pred = [model.predict_proba(X[ix, :])[:, 1] for ix in idx]
pfeats = map(get_feat, pred)

pfeats = np.array(pfeats)
turget = np.array(turget)

tsmodel = TSNE(n_components=2, verbose=True)
Y = tsmodel.fit_transform(pfeats)

import matplotlib.pyplot as plt
plt.scatter(Y[:, 0], Y[:, 1], c=turget)
plt.axis('off')
plt.axis('tight')
plt.show()


from sklearn import svm
clf = svm.SVC(C=1e5, verbose=True,probability=True)
clf.fit(pfeats, turget)

print confusion_matrix(turget, clf.predict(pfeats))
print accuracy_score(turget, clf.predict(pfeats))
print roc_auc_score(turget, clf.predict_proba(pfeats)[:,1])


F, T = chilly_build(2)
pbuilds = clf.predict_proba(F)

sdicts = {'f':T, 'pbuilds': pbuilds}
np.save('p3_80', sdicts)

p180 = np.load('p1_80.npy').item()
p280 = np.load('p2_80.npy').item()
p380 = np.load('p3_80.npy').item()

import pandas as pd

df = pd.DataFrame()
df['File'] = np.append(np.append(p180['f'], p280['f']), p380['f'])
df['Class'] = np.append(np.append(p180['pbuilds'][:, 1], p280['pbuilds'][:, 1]), p380['pbuilds'][:, 1])
df.sort_values(['File'])

en = pd.read_csv('./ensemble.csv')
en.sort_values(['File'])

df_en = pd.DataFrame()
df_en['File'] = en['File']
df_en['Class'] = (en.Class + df.Class)/2.
df_en.to_csv('please_be_74.csv', index=False)

def warp(x):
    """
        Logarithmically rescale values with p < 0.1 and (1-p) < 0.1 to avoid loss of precision
    """
    x = np.copy(x)
    low = x<0.1
    high = x>0.35

    # xl = x[low]
    # pmin = np.log10(xl[xl!=0].min())-1
    # lx = np.log10(xl)
    # lx[np.isinf(lx)] = pmin
    # lx = (lx+1)*(3./(lx.max()-lx.min()))-1
    # xl = 10**lx
    # x[low] = xl

    xh = x[high]
    pmin = np.log10(1-xh[xh!=1].max())-1
    lx = np.log10(1-xh)
    lx[np.isinf(lx)] = pmin
    lx = (lx+1)*(1./(lx.max()-lx.min()))-1
    xh = 1-10**lx
    x[high] = xh

    return x





en_fl = en[en.File.str.match('new_1_*|new_3_*')]

en_nw = pd.DataFrame()
en_nw['File'] = p280['f']
en_nw['Class'] = p280['pbuilds'][:, 1]

en_fn = en_nw.append(en_fl)
en_fn.to_csv('this_is_sparta.csv', index=False)
