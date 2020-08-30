from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from dataset import *
import pandas as pd
import sys
import numpy as np
import pickle
import time


class MyClassifier:
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name
        self.info = {'name': name}


class MyKNN(MyClassifier):
    def __init__(self, name, estimator=KNeighborsClassifier()):
        super().__init__(estimator=estimator, name=name)
        self.info['n_neighbors'] = self.estimator.n_neighbors


class MySVM(MyClassifier):
    def __init__(self, name, estimator=SVC()):
        super().__init__(estimator=estimator, name=name)
        kernel = self.estimator.get_params()['kernel']
        self.info['kernel'] = kernel
        self.info['C'] = self.estimator.get_params()['C']
        if kernel != 'linear':
            self.info['gamma'] = self.estimator.get_params()['gamma']



def my_train_test_split(dataset):
    X = dataset.features.X
    y = dataset.features.y.ravel()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def get_model_file_path(classifier, dataset):
    filename = '{}.sav'.format(dataset.name.lower())
    path_to_file = os.path.join('models', classifier.name, filename)
    return path_to_file


def find_best_and_save_one(classifier, dataset, scoring):
    folder = os.path.join('models', classifier.name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = get_model_file_path(classifier, dataset)
    if os.path.exists(path_to_file):
        raise FileExistsError('File {} already exists!'.format(path_to_file))
    X_train, X_test, y_train, y_test = my_train_test_split(dataset)
    grid = GridSearchCV(classifier.estimator, classifiers_parameters[classifier],
                        cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    if isinstance(classifier, MySVM):
        best_classifier = MySVM(name=classifier.name, estimator=grid.best_estimator_)
    elif isinstance(classifier, MyKNN):
        best_classifier = MyKNN(name=classifier.name, estimator=grid.best_estimator_)
    else:
        raise ValueError('Error!')
    pickle.dump(best_classifier, open(path_to_file, 'wb'))
    print('Model saved to {}'.format(path_to_file))
    return best_classifier


def find_best_and_save_all():
    for classifier in [knn, svm_linear, svm_nonlinear]:
        print('WORKING WITH CLASSIFIER {}'.format(classifier.name.upper()))
        for dataset in datasets_list_original + datasets_list_negative_binary:
            print('WORKING WITH DATASET {}'.format(dataset.name.upper()))
            try:
                find_best_and_save_one(classifier, dataset, scoring=scoring)
            except FileExistsError:
                print('File {} already existed!'.format(get_model_file_path(classifier, dataset)))
                continue


class Model:
    def __init__(self, classifier, dataset):
        pass




svm_linear = MySVM('SVM_linear')
svm_nonlinear = MySVM('SVM_nonlinear')
knn = MyKNN('KNN')
scoring = 'accuracy'
svm_nonlinear_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]},
    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]}]
svm_linear_parameters = {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]}
knn_parameters = {'n_neighbors': np.arange(2, 100)}
classifiers_parameters = {svm_linear: svm_linear_parameters,
                          svm_nonlinear: svm_nonlinear_parameters,
                          knn: knn_parameters}

if __name__ == '__main__':
    find_best_and_save_all()
