from dataset import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
emodb = Dataset('Emo-DB', EMODB_FOLDER, 'German')
ravdess = Dataset('Ravdess', RAVDESS_FOLDER, 'English')
cremad = Dataset('Crema-D', CREMAD_FOLDER, 'English')
savee = Dataset('SAVEE', SAVEE_FOLDER, 'English')
tess = Dataset('TESS', TESS_FOLDER, 'English')

datasets = [iemo, emodb, ravdess, cremad, savee, tess]  
for dataset in datasets:
    X = dataset.features.X
    y = dataset.features.y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    print('\nWORKING WITH DATASET {}...'.format(dataset.name))
    print('\n{} training samples; {} test samples.'.format(len(y_train), len(y_test)))
    n_neighbors_list = np.arange(2, 300, 2)
    train_score_list = []
    test_score_list = []
    _n_neighbors_list = []
    for k in n_neighbors_list:
        try:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            train_score = knn.score(X_train, y_train)
            test_score = knn.score(X_test, y_test)
            train_score_list.append(train_score)
            test_score_list.append(test_score)
            _n_neighbors_list.append(k)
            print('K-NN with {} neighbors, train score: {}, test score: {}'.format(k, train_score, test_score))
        except ValueError:
            print('Value Error! Breaking...')
            break
    print('Writing to file...')
    df = pd.DataFrame(data=[train_score_list, test_score_list], index=['train_scores', 'test_scores'], columns=_n_neighbors_list)
    df.to_csv('classifiers\\knn\\{}_train_test_score.csv'.format(dataset.name.lower()))
    print('Job done!')




