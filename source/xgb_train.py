import numpy as np
import pandas as pd
from os.path import basename

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style='whitegrid', palette='Paired')

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import xgboost as xgb

RANDOM_STATE = 840

# sorry, but I have to keep this global variable
global_scaler = MinMaxScaler()

def load_safe_list(LIST_PATH):
    df = pd.read_csv(LIST_PATH)
    df_safe = df[df['safe'] == 1]
    return list(df_safe['image'])

def scale_data(data, phase='train'):
    if phase == 'train':
        return global_scaler.fit(data).transform(data)
    else:
        return global_scaler.transform(data)

def load_train_dataset(train_path, list_path='', scale_option=True):
    data = np.load(train_path).item()
    file_name = np.array(map(lambda x: basename(x), data['mat_files']))
    train_data = np.array(data['data'])
    label = np.array(data['target'])
    if list_path:
        safe_list = load_safe_list(list_path)
        trindx = map(lambda x: x in safe_list, file_name)
        safe_indx = np.where(trindx)
        train_data = train_data[safe_indx]
        label = label[safe_indx]
    if scale_option == True:
        train_data = scale_data(train_data, phase='train')
    return train_data, label

def load_test_dataset(test_path, scale_option=True):
    data = np.load(test_path).item()
    file_name = np.array(map(lambda x: basename(x), data['mat_files']))
    file_name = np.reshape(file_name, (file_name.shape[0] / 30, 30))
    file_name = map(lambda x: x[0], file_name)
    test_data = data['data']
    if scale_option == True:
        test_data = scale_data(test_data, phase='test')
    return test_data, file_name

def split_data(data, label, sample=True):
    if sample == True:
        X_omit, X_select, y_omit, y_select = train_test_split(
            data, label, test_size=4200, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X_select, y_select, test_size=1200, random_state=RANDOM_STATE)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, test_size=0.33, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

def fit_xgb(X_train, X_test, y_train, y_test, full_set, pred_set=None):
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    xg_full = xgb.DMatrix(full_set)
    if pred_set is not None:
        xg_pred = xgb.DMatrix(pred_set)
    # specify parameters via map
    label = xg_train.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    print 'adjust label weight to ' + str(ratio)
    params = {
    "objective": "binary:logistic",
    "booster" : "gbtree",
    "eval_metric": "auc",
    "eta": 0.1,
    'scale_pos_weight': ratio,
    "tree_method": 'auto',
    "max_depth": 10,
    "silent": 0,
    "seed": RANDOM_STATE,
    }
    num_boost_round = 50
    early_stopping_rounds = 100
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 150
    bst = xgb.train(params, xg_train, num_boost_round, evals=watchlist,
                early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    if pred_set is None:
        return bst.predict( xg_test ), bst.predict( xg_full )
    else:
        return bst.predict( xg_test ), bst.predict( xg_full ), bst.predict( xg_pred )

def fit_svc(X_train, X_test, y_train, y_test, full_set, pred_set=None):
    clf = SVC(C=1e6, kernel='rbf', gamma=0.02, shrinking=True, probability=True,
                class_weight='balanced', random_state=0).fit(X_train, y_train)
    if pred_set is None:
        return clf.predict_proba(X_test)[:, 1], clf.predict_proba(full_set)[:, 1]
    else:
        return clf.predict_proba(X_test)[:, 1], clf.predict_proba(full_set)[:, 1], clf.predict_proba(pred_set)[:, 1]

def estimate_result(y_test, prediction):
    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

def longest_consecutive(L):
    max_count = 0
    i = 0
    while i < len(L):
        count = 0
        if L[i] == 1:
            j = i
            while j < len(L):
                if L[j] == 1:
                    count += 1
                else:
                    break
                j += 1
            i = j
        else:
            i += 1    
        max_count = max(count, max_count)
    return max_count

def window_feature(vector):
    v_mean = vector.mean()
    v_max = vector.max()
    v_min = vector.min()
    v_var = np.var(vector)
    binary_vector = map(lambda x: 1 if x > 0.5 else 0, vector)
    count = sum(binary_vector)
    longest = longest_consecutive(binary_vector)
    return np.array([v_mean, v_max, v_min, v_var, count, longest])


def generate_window_dataset(prediction, label):
    row_num = label.shape[0] / 30
    new_data = prediction.reshape((row_num, 30))
    new_label = label.reshape((row_num, 30)).mean(axis=1)
    new_data = np.apply_along_axis(window_feature, 1, new_data)
    return new_data, new_label

def generate_window_test(prediction, file_name):
    row_num = prediction.shape[0] / 30
    new_data = prediction.reshape((row_num, 30))
    new_data = np.apply_along_axis(window_feature, 1, new_data)
    return new_data, filename

def train_patient(NO):
    list_path = '/home/ruifan/dataset/seizure/labels_safe.csv'
    train_path = '/home/ruifan/dataset/seizure/new_feature/' + str(NO) + '_train.npy'
    test_path = '/home/ruifan/dataset/seizure/new_feature/' + str(NO) + '_test.npy'

    print 'now build the train set'
    train_data, label = load_train_dataset(train_path, list_path)
    X_train, X_test, y_train, y_test = split_data(train_data, label, sample=False)
    print 'data loaed'
    print 'train size: ' + str(y_train.shape)
    print '0 label_num: ' + str(np.sum(y_train == 0)) + '  1 label_num: ' + str(np.sum(y_train == 1))
    print 'test size: ' + str(y_test.shape)
    print '0 label_num: ' + str(np.sum(y_test == 0)) + '  1 label_num: ' + str(np.sum(y_test == 1))
    
    print 'now build the submission set'
    pred_data, file_name = load_test_dataset(test_path)
    print 'data loaed'
    print 'submission size: ' + str(pred_data.shape)
    
    ensemble_prediction = []
    full_prediction = []
    submission = []
    
    print 'start to fit XGB'
    xg_prediction, xg_full, xg_submission = fit_xgb(X_train, X_test, y_train, y_test, train_data, pred_data)
    ensemble_prediction.append(xg_prediction)
    full_prediction.append(xg_full)
    submission.append(xg_submission)

    print 'start to fit SVC'
    svc_prediction, svc_full, svc_submission = fit_svc(X_train, X_test, y_train, y_test, train_data, pred_data)
    ensemble_prediction.append(svc_prediction)
    full_prediction.append(svc_full)
    submission.append(svc_submission)

    ensemble_prediction = np.array(ensemble_prediction).mean(axis=0)
    full_prediction = np.array(full_prediction).mean(axis=0)
    submission = np.array(submission).mean(axis=0)
    
    estimate_result(y_test, xg_prediction)
    estimate_result(y_test, svc_prediction)
    estimate_result(y_test, ensemble_prediction)
    plt.show()
    
    print 'confusion matrix for test'
    print confusion_matrix(y_test, map(lambda x: 0 if x < 0.5 else 1, ensemble_prediction))
    
    estimate_result(label, xg_full)
    estimate_result(label, svc_full)
    estimate_result(label, full_prediction)
    plt.show()
    
    print 'confusion matrix for FULL SET'
    print confusion_matrix(label, map(lambda x: 0 if x < 0.5 else 1, full_prediction))
    
    #########################################################
    print 'now generate new training set'
    new_data, new_label = generate_window_dataset(full_prediction, label)
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, 
                                test_size=0.33, random_state=RANDOM_STATE)
    print 'data loaed'
    print 'train size: ' + str(y_train.shape)
    print '0 label_num: ' + str(np.sum(y_train == 0)) + '  1 label_num: ' + str(np.sum(y_train == 1))
    print 'test size: ' + str(y_test.shape)
    print '0 label_num: ' + str(np.sum(y_test == 0)) + '  1 label_num: ' + str(np.sum(y_test == 1))
    
    print 'now generate the submission set'
    new_submission, new_file_name = generate_window_test(submission, file_name)
    
    print 'start to fit XGB on window feature'
    xg_prediction, xg_full, xg_submission = fit_xgb(X_train, X_test, y_train, y_test, new_data, new_submission)
    estimate_result(y_test, xg_prediction)
    plt.show()
    print 'confusion matrix for test'
    print confusion_matrix(y_test, map(lambda x: 0 if x < 0.5 else 1, xg_prediction))
    
    estimate_result(y_test, xg_prediction)
    plt.show()
    print 'confusion matrix for FULL SET'
    print confusion_matrix(new_label, map(lambda x: 0 if x < 0.5 else 1, xg_full))
    
    return xg_submission, new_file_name