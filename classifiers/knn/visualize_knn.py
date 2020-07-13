import pandas as pd
import matplotlib.pyplot as plt
import os


def visualize_train_test():
    csv_files = []
    for file in os.listdir():
         if file.endswith('raw.csv'):
        #  if 'standardized.csv' in file and 'assembly' in file:
            #  if file.endswith('normalized.csv'):
            csv_files.append(file)
    for csv_file in csv_files:
        dataset_name = csv_file.split('_')[0].upper()
        df = pd.read_csv(csv_file, sep=';')
        n_neighbors, train_score, test_score = df['n_neighbors'], df['train_score'], df['test_score']
        n_neighbors = n_neighbors.values.tolist()
        train_score = train_score.values.tolist()
        test_score = test_score.values.tolist()
        _ = df.iloc[df['test_score'].argmax()].values.tolist()
        max_test_score_n = _[0]
        max_test_score = _[2]

        plt.figure(figsize=[12.8, 10.24])
        plt.scatter(max_test_score_n, max_test_score,
                    label='Max test score {:.3f} with {:.0f} neighbors'.format(max_test_score, max_test_score_n),
                    c='red', marker=7, s=500, zorder=10)
        plt.plot(n_neighbors, train_score,
                 label='Train', linewidth=7.0)
        plt.plot(n_neighbors, test_score,
                 label='Test', linewidth=7.0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xscale('linear')
        plt.xlabel('Number of neighbors', fontsize=30)
        plt.ylabel('Model accuracy', fontsize=30)
        plt.title('{} k-NN multiclass classification \n(raw features)'.format(dataset_name), fontsize=40)
        plt.legend(fontsize=20)
        plt.savefig(fname='{}_plot_raw.png'.format(dataset_name.lower()))
        print('Saved {} plot successfully!'.format(dataset_name))


def visualize_raw_vs_norm_standard(names_list):
    for name in names_list:
        print(name)
        for file in os.listdir():
            if file.endswith('.csv') and (name in file):
                if 'standardized' in file:
                    df = pd.read_csv(file, sep=';')
                    n_neighbors = df['n_neighbors'].values.tolist()
                    test_score_standardized = df['test_score'].values.tolist()
                    _ = df.iloc[df['test_score'].argmax()].values.tolist()
                    max_test_score_n = _[0]
                    max_test_score = _[2]
                if 'raw' in file:
                    df = pd.read_csv(file, sep=';')
                    test_score_raw = df['test_score'].values.tolist()
                if 'normalized' in file:
                    df = pd.read_csv(file, sep=';')
                    test_score_normalized = df['test_score'].values.tolist()
        plt.figure(figsize=[12.8, 10.24])
        plt.scatter(max_test_score_n, max_test_score,
                    label='Max test score {:.3f} with {:.0f} neighbors'.format(max_test_score, max_test_score_n),
                    c='red', marker=7, s=500, zorder=10)
        plt.plot(n_neighbors, test_score_raw,
                 label='Raw features', linewidth=7.0)
        plt.plot(n_neighbors, test_score_standardized,
                 label='Standardized features', linewidth=7.0)
        plt.plot(n_neighbors, test_score_normalized,
                 label='Normalized features', linewidth=7.0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xscale('linear')
        plt.xlabel('Number of neighbors', fontsize=30)
        plt.ylabel('Model accuracy', fontsize=30)
        plt.title('{} k-NN multiclass classification '.format(name.upper()), fontsize=40)
        plt.legend(fontsize=20)
        plt.savefig(fname='{}_plot_raw_vs_norm_standard.png'.format(name.lower()))
        print('Saved {} plot successfully!'.format(name))




if __name__ == '__main__':
    pass


