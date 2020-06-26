from dataset import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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


if __name__ == '__main__':
    emodb = Dataset('Emo-DB', EMODB_FOLDER, 'German')
    iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
    ravdess = Dataset('Ravdess', RAVDESS_FOLDER, 'English')
    cremad = Dataset('Crema-D', CREMAD_FOLDER, 'English')
    savee = Dataset('SAVEE', SAVEE_FOLDER, 'English')
    tess = Dataset('TESS', TESS_FOLDER, 'English')
    datasets_list = [emodb, iemo, ravdess, cremad, savee, tess]
    knn_multi_classification(datasets_list)