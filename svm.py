from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

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

    def get_scores(self, dataset):
        X_train, X_test, y_train, y_test = my_train_test_split(dataset)
        y_pred = self.estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        prec = precision_score(y_test, y_pred, average='micro')
        rec = recall_score(y_test, y_pred, average='micro')
        return [acc, prec, rec, f1]


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

class MyModel:
    def __init__(self, filename):
        path = os.path.join('models', filename)
        self.file = filename
        classifier_class = pickle.load(open(path, 'rb'))
        self.classifier = classifier_class
        dataset_name = filename.split('_')[-1].split('.')[0]
        for dataset in datasets_list_original + datasets_list_negative_binary:
            if dataset.name.lower() == dataset_name:
                self.dataset = dataset
        acc, prec, rec, f1 = classifier_class.get_scores(self.dataset)
        _info = {
            'dataset': self.dataset.name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        self.info = {**_info, **classifier_class.info}



def my_train_test_split(dataset):
    X = dataset.features.X
    y = dataset.features.y.ravel()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def get_model_file_path(classifier, dataset):
    filename = '{}_{}.sav'.format(classifier.name, dataset.name.lower())
    path_to_file = os.path.join('models', filename)
    return path_to_file


def find_best_and_save_one(classifier, dataset, scoring):
    folder = 'models\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = get_model_file_path(classifier, dataset)
    print(path_to_file)
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


def create_summary_file():
    dfs_list = []
    folder = MODELS_FOLDER
    model_files_list = [file for file in os.listdir(folder) if file.endswith('.sav')]
    for model_file in model_files_list:
        mdl = MyModel(model_file)
        print(mdl.dataset.name, ' ', mdl.classifier.name)
        df = pd.DataFrame(mdl.info, index=[0])
        dfs_list.append(df)
    final_df = pd.concat(dfs_list, axis=0, ignore_index=True)
    final_df.to_csv('models\\models_summary.csv', sep=';', index=False)


def create_classification_report_file():
    dfs_list = []
    folder = MODELS_FOLDER
    model_files_list = [file for file in os.listdir(folder) if file.endswith('.sav')]
    for model_file in model_files_list:
        mdl = MyModel(model_file)
        print(mdl.dataset.name, ' ', mdl.classifier.name)
        dataset = mdl.dataset
        X_train, X_test, y_train, y_test = my_train_test_split(dataset)
        y_pred = mdl.classifier.estimator.predict(X_test)
        report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
        keys = list(report_dict.keys())
        keys.remove('accuracy')
        dictionary = {
            'class': [],
            'precision': [],
            'recall': [],
            'f1-score': []
        }
        for key in keys:
            dictionary['class'].append(key)
            dictionary['precision'].append(report_dict[key]['precision'])
            dictionary['recall'].append(report_dict[key]['recall'])
            dictionary['f1-score'].append(report_dict[key]['f1-score'])
        df = pd.DataFrame(dictionary)
        df['classifier'] = mdl.classifier.name
        df['dataset'] = mdl.dataset.name
        dfs_list.append(df)
        final_df = pd.concat(dfs_list, axis=0, ignore_index=True)
        folder = 'models\\classification_reports\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
        final_df.to_csv(folder + 'summarized_classification_reports.csv', sep=';', index=False)


def split_report_file():
    for dataset in datasets_list_original + datasets_list_negative_binary:
        print(dataset.name)
        for classifier in ['KNN', 'SVM_linear', 'SVM_nonlinear']:
            df = pd.read_csv('models\\classification_reports\\summarized_classification_reports.csv', delimiter=';')
            print(classifier)
            df = df.loc[(df['dataset'] == dataset.name) & (df['classifier'] == classifier)]
            df = df.drop(['classifier', 'dataset'], axis=1)
            df.to_csv('models\\classification_reports\\{}_{}_classification_report.csv'
                      .format(classifier, dataset.name.lower()), sep=';', index=False)




svm_linear = MySVM('SVM-linear')
svm_nonlinear = MySVM('SVM-nonlinear')
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
    split_report_file()




































