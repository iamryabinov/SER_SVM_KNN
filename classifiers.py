from dataset import *
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class Classifier:
    def __init__(self, results_folder='classifiers\\'):
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)


class KNearestNeighbors(Classifier):
    def __init__(self, name='knn'):
        super().__init__()
        self.classifier_name = name
        self.results_folder = self.results_folder + name + '\\'
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

class KNNBaseMethod(KNearestNeighbors):
    def __init__(self, method_name='base'):
        super().__init__()
        self.method_name = method_name

    def classify(self, dataset, n, preprocess='false'):
        X = dataset.features.X
        y = dataset.features.y.ravel()
        if preprocess == 'normalize':
            X = Normalizer().fit_transform(X)
            suffix = '_normalized'
        elif preprocess == 'standardize':
            X = StandardScaler().fit_transform(X)
            suffix = '_standardized'
        elif preprocess == 'false':
            suffix = '_raw'
        else:
            raise ValueError('Unknown value for preprocess: should be either "normalize" or "standardize" or "false"!')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        try:
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X_train, y_train)
            train_score = knn.score(X_train, y_train)
            test_score = knn.score(X_test, y_test)
            print('K-NN with {} neighbors, train score: {}, test score: {}'.format(n, train_score, test_score))
        except ValueError:
            raise ValueError('Too many neighbors!')
        data = {'n_neighbors': n,
                'train_score' + suffix: train_score,
                'test_score' + suffix: test_score
                }
        return data

    def evaluate(self, dataset, n_from=2, n_to=75, preprocess='false'):
        # Preprocess
        X = dataset.features.X
        y = dataset.features.y.ravel()
        results_file = f'{self.results_folder}{dataset.name.lower()}_{self.method_name}_train-test-score'
        if preprocess == 'normalize':
            X = Normalizer().fit_transform(X)
            suffix = '_normalized'
        elif preprocess == 'standardize':
            X = StandardScaler().fit_transform(X)
            suffix = '_standardized'
        elif preprocess == 'false':
            suffix = '_raw'
        else:
            raise ValueError('Unknown value for preprocess: should be either "normalize" or "standardize" or "false"!')
        results_file = results_file + suffix + '.csv'
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        print(f'\nWORKING WITH DATASET {dataset.name.upper()}, PREROCESS = {preprocess.upper()}')
        print('\n{} training samples; {} test samples.'.format(len(y_train), len(y_test)))
        n_neighbors_list = np.arange(n_from, n_to + 1)
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
                'train_score' + suffix: train_score_list,
                'test_score' + suffix: test_score_list
                }
        results = pd.DataFrame(data, columns=['n_neighbors', 'train_score' + suffix, 'test_score' + suffix])
        results.to_csv(results_file, index=False, sep=';')
        print('Job done!')

    def summarize_results(self, dataset):
        print('WORKING WITH {} METHOD ON DATASET {}'.format(self.method_name.upper(), dataset.name.upper()))
        csv_files = []
        for file in os.listdir(self.results_folder):
            if file.endswith('.csv') and dataset.name.lower() in file and self.method_name.lower() in file:
                csv_files.append(self.results_folder + file)
        results = {
            'raw': '',
            'normalized': '',
            'standardized': ''
        }
        for csv_file in csv_files:
            suffix = csv_file.split('_')[-1].split('.')[0]
            df = pd.read_csv(csv_file, delimiter=';')
            results[suffix] = df
        _df = pd.merge(results['raw'], results['normalized'], on='n_neighbors')
        summary = pd.merge(_df, results['standardized'], on='n_neighbors')
        filename = f'{self.results_folder}{dataset.name.lower()}_{self.method_name}_train_test_score_summary.csv'
        summary.to_csv(filename, sep=';', index=False)

    def get_best_result(self, dataset):
        best_results_dict = {
            'preprocessing': ['raw', 'normalize', 'standardize'],
            'n_neighbors': [],
            'test_score': []
        }
        for file in os.listdir(self.results_folder):
            if dataset.name.lower() in file and self.method_name.lower() in file and 'summary' in file:
                filename = file
        print(filename)
        df = pd.read_csv(self.results_folder + filename, delimiter=';')
        for preprocessing in ['raw', 'normalized', 'standardized']:
            best_of_suffix = df.iloc[df['test_score_{}'.format(preprocessing)].argmax()]
            best_k = best_of_suffix['n_neighbors']
            best_score = (best_of_suffix['test_score_{}'.format(preprocessing)])
            best_results_dict['n_neighbors'].append(best_k)
            best_results_dict['test_score'].append(best_score)
        best_results = pd.DataFrame(best_results_dict)
        return best_results
        pass



def knn_confusion_matrix(dataset, n):
    directory = 'plots\\knn\\confusion_matrix\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    X = dataset.features.X
    y = dataset.features.y.ravel()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    disp = plot_confusion_matrix(
        knn, X_test, y_test,
        normalize='true', xticks_rotation=30, cmap=plt.cm.Oranges,
    )
    disp.ax_.set_title('{} k-NN confusion matrix'.format(dataset.name))
    plt.show()
    # filename = directory + '{}_knn_confusion_matrix.png'.format(dataset.name)
    # plt.savefig(fname=filename)
    # print('Saved {}'.format(filename))
    # plt.close()


if __name__ == '__main__':
    knn = KNNBaseMethod('first_try')
    print(knn.get_best_result(all_english_six_pos_neg_neu))
    knn_confusion_matrix(all_english_six_pos_neg_neu, 11)
