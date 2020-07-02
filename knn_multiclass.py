from dataset import *
from sklearn.preprocessing import StandardScaler, Normalizer  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def knn_multi_classification(datasets_list):
    for dataset in datasets_list:
        X = dataset.features.X
        y = dataset.features.y.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, stratify=y)
        print('\nWORKING WITH DATASET {}...'.format(dataset.name))
        print('\n{} training samples; {} test samples.'.format(len(y_train), len(y_test)))
        n_neighbors_list = np.arange(2, 101)
        n_neighbors_list_fact = []
        train_score_list = []
        test_score_list = []
        for k in n_neighbors_list:
            try:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                train_score = knn.score(X_train, y_train)
                test_score = knn.score(X_test, y_test)
                train_score_list.append(train_score)
                test_score_list.append(test_score)
                n_neighbors_list_fact.append(k)
                print('K-NN with {} neighbors, train score: {}, test score: {}'.format(k, train_score, test_score))
            except ValueError:
                print('Value Error! Too many neighbors! Breaking...')
                break
        print('Done! Max test score {}'.format(max(test_score_list)))
        print('Writing to file...')
        data = {'n_neighbors': n_neighbors_list_fact, 
                'train_score': train_score_list, 
                'test_score': test_score_list
                }
        df = pd.DataFrame(data, columns=['n_neighbors', 'train_score', 'test_score'])
        df.to_csv('classifiers\\knn\\{}_train_test_score.csv'.format(dataset.name.lower()), index=False, sep=';')
        print('Job done!')



def standardize(datasets_list):
    for dataset in datasets_list:
            dataset.name = dataset.name + '_standardized'
            dataset.features.X = StandardScaler().fit_transform(dataset.features.X)
            print('Standardization complete!')

def normalize(datasets_list):
    for dataset in datasets_list:
            dataset.name = dataset.name + '_normalized'
            dataset.features.X = Normalizer().fit_transform(dataset.features.X)
            print('Normalization complete!')

def knn_confusion_matrix(dataset, n):
    directory = 'plots\\knn\\confusion_matrix\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    dataset.features.X = StandardScaler().fit_transform(dataset.features.X)
    X = dataset.features.X
    y = dataset.features.y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    disp = plot_confusion_matrix(
        knn, X_test, y_test, 
        normalize='true', xticks_rotation=30, cmap=plt.cm.Oranges,
        )
    disp.ax_.set_title('{} k-NN confusion matrix'.format(dataset.name))
    filename = directory + '{}_knn_confusion_matrix.png'.format(dataset.name)
    plt.savefig(fname=filename)
    print('Saved {}'.format(filename))
    plt.close()
       


if __name__ == '__main__':
    knn_multi_classification([all_english_six])
    standardize([all_english_six])
    knn_multi_classification([all_english_six])