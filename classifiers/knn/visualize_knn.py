import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = []
if __name__ == '__main__':
    for file in os.listdir():
        if not file.endswith('.csv'):
            continue
        csv_files.append(file)

    for csv_file in csv_files:
        x_max = 0
        dataset_name = csv_file.split('_')[0].upper()
        df = pd.read_csv(csv_file)
        test_scores = df.iloc[[1]].drop('Unnamed: 0', axis=1)
        max_test_score = test_scores.values.max()
        x_max = int(test_scores.idxmax(axis=1).values.tolist()[0])
        x = df.columns.values[1:]
        y1 = df.iloc[0].values[1:]
        y2 = df.iloc[1].values[1:]
        plt.figure(figsize=[12.8, 10.24])
        plt.plot(x, y1, label='Train')
        plt.plot(x, y2, label='Test')
        plt.scatter(x_max, max_test_score,
                    label='Max test score {:.4f} with {} neighbors'.format(max_test_score, x_max), c='green', marker=7,
                    s=150)
        plt.xscale('linear')
        plt.xlabel('Number of neighbors', fontsize=12)
        plt.ylabel('Model accuracy', fontsize=12)
        plt.title('{} k-NN multiclass classification'.format(dataset_name), fontsize=20)
        plt.legend(fontsize=12)
        plt.show()
        plt.close()

