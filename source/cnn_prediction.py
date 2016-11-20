import utils as iu
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
import csv
import os.path

from sklearn.model_selection import train_test_split


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


def build_model(inp_shape):
    model = Sequential()

    model.add(
        Convolution2D(
            32, 3, 3, border_mode='valid', input_shape=inp_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

    return model


def fit_model(model, data, batch_size=32, nb_epoch=75):
    X_train, Y_train, X_test, Y_test = data
    model.fit(X_train,
              Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              shuffle=True,
              validation_data=(X_test, Y_test))


def write_submission(run_id):

    results = [predict_for_patient(i, run_id) for i in range(3)]

    data = [['File', 'Class']]
    for res in results:
        for fn, c in res:
            data.append([fn, c])

    with open('results_r{0}.csv'.format(run_id), 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)


def predict_for_patient(patient_id, run_id):

    print "predicting ", patient_id + 1
    data = iu.load_data_for_patient(patient_id, dtype='test')
    # spec_data = iu.load_data_for_patient(
    #     patient_id, dtype='test', file_name='spectrograms_4x4.npy')
    spec_data = np.array([
        iu.stitch_spectrograms_images(spec, (16, 1))
        for spec in data['raw_spectrograms']
    ])
    X = np.array([convert_to_image_format(rd) for rd in spec_data])
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    model = load_model(get_model_file(run_id, patient_id))
    predictions = model.predict_classes(X)
    return zip(data['file_name'], predictions)


def get_model_file(run_id, patient_id):

    return os.path.join(iu.output_directory, "keras_model_run{0}_p{1}".format(
        run_id, patient_id))


def generate_model(run_id, patient_id, model_gen_func=build_model):
    data = iu.load_data_for_patient(patient_id)
    safe_indexes = iu.get_safe_index(data['file_name'])

    data = apply_indexes(data, safe_indexes)
    # spec_4x4 = iu.load_data_for_patient(
    #     patient_id, file_name='spectrograms_4x4.npy')[safe_indexes]
    spec_16x1 = np.array([
        iu.stitch_spectrograms_images(spec, (16, 1))
        for spec in data['raw_spectrograms']
    ])
    X = np.array([convert_to_image_format(rd) for rd in spec_16x1])
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    Y = np_utils.to_categorical(data['target'], 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    print "Building model ...", patient_id
    model = model_gen_func(X_train[0].shape)

    print "Fitting model ...", patient_id
    fit_model(model, (X_train, Y_train, X_test, Y_test))
    model.save(get_model_file(run_id, patient_id))


run_id = 3
generate_model(run_id, 0)
generate_model(run_id, 1)
generate_model(run_id, 2)
# write
write_submission(run_id)
