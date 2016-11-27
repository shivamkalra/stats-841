import utils as ut
import numpy as np

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


def apply_indexes(data, indexes):
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k][indexes]

    return new_data


def apply_safe_indexes(data):
    safe_indexes = ut.get_safe_index(data['file_name'])
    return apply_indexes(data, safe_indexes)


def build_model():
    model = Sequential()

    model.add(
        Convolution3D(
            32,
            3,
            2,
            4,
            border_mode='valid',
            dim_ordering='th',
            input_shape=(1, 16, 6, 60),
            init='he_normal'))

    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(MaxPooling3D(pool_size=(2, 1, 3), dim_ordering='th'))

    model.add(Convolution3D(64, 2, 1, 3, dim_ordering='th', init='he_normal'))
    model.add(PReLU())
    model.add(MaxPooling3D(pool_size=(2, 1, 3), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))

    model.add(Dense(100, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Dense(2, init='he_normal'))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model


def generate_x_data(data, indexes=None):
    if indexes is None:
        indexes = np.arange(data['raw_spectrograms'].shape[0])

    return np.array(
        [[im] for im in data['raw_spectrograms'][indexes, :, :, :]])


def generate_x_y_data(data, indexes=None):

    if indexes is None:
        indexes = np.arange(data['target'].shape[0])

    X = generate_x_data(data, indexes)
    Y = np_utils.to_categorical(data['target'][indexes], 2)

    return X, Y


def data_generator(data, shuffle=True, batch_size=32):

    dlen = data['target'].shape[0]
    number_of_batches = np.ceil(dlen / batch_size)
    counter = 0
    sample_index = np.arange(dlen)

    if shuffle:
        np.random.shuffle(sample_index)

    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter +
                                                                      1)]

        yield generate_x_y_data(data, batch_index)

        counter += 1
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def predict(patient_id):

    tr_d = apply_safe_indexes(
        ut.load_data_for_patient(
            patient_id=0, dtype='train'))
    te_d = ut.load_data_for_patient(patient_id=0, dtype='test')

    te_X = generate_x_data(te_d)

    nbags = 5
    nfolds = 5
    kf = KFold(n_splits=nfolds)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    pred = np.zeros((te_d['raw_spectrograms'].shape[0], ))
    for train_index, val_index in kf.split(tr_d['target']):

        tr_tr_d = apply_indexes(tr_d, train_index)
        val_d = apply_indexes(tr_d, val_index)

        for _ in range(nbags):

            model = build_model()

            dg = data_generator(tr_tr_d)
            model.fit_generator(
                dg,
                samples_per_epoch=tr_tr_d['target'].shape[0],
                validation_data=generate_x_y_data(val_d),
                callbacks=[early_stopping],
                nb_epoch=13)

            pred += model.predict_proba(te_X)[:, 1]

    pred /= 1. * nbags * nfolds

    return te_d['file_name'], pred


run_id = 11

p1 = predict(patient_id=0)
p2 = predict(patient_id=1)
p3 = predict(patient_id=2)

data = [['File', 'Class']]
for res in [p1, p2, p3]:
    for fn, c in res:
        data.append([fn, c])

with open('results_r{0}.csv'.format(run_id), 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)
