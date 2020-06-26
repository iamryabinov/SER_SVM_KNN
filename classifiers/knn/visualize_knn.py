import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = []
if __name__ == '__main__':
    for file in os.listdir():
        if file.endswith('.csv'):
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
        plt.title('{} k-NN multiclass classification'.format(dataset_name), fontsize=40)
        plt.legend(fontsize=20)
        plt.savefig(fname='{}_plot.png'.format(dataset_name.lower()))
        print('Saved {} plot successfully!'.format(dataset_name))

