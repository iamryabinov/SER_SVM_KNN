from dataset import *
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def plot_tsne(dataset):
    print('\nWORKING WITH {}'.format(dataset.name))
    dataset.name = dataset.name.split('_')[0].lower()
    X, y = dataset.features.X, dataset.features.y.ravel()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)
    if len(y) > 2500:
        X = X_test
        y = y_test
        print('Fetched data')
    hue_order = np.sort(dataset.features.contents.label.unique()).tolist()
    palette = {'ang': '#FF0000',
               'hap': '#23FF00',
               'sad': '#3D00FF',
               'neu': '#5D5D5D',
               'dis': '#A400FF',
               'fea': '#00FFFF',
               'sur': '#00A700',
               'cal': '#A78A00',
               'exc': '#FFD300',
               'fru': '#FF7400',
               'bor': '#A78A00'
               }
    markers = {'ang': 'o',
               'hap': 's',
               'sad': 'P',
               'neu': 'X',
               'dis': '^',
               'fea': 'v',
               'sur': '*',
               'cal': 'p',
               'exc': 'D',
               'fru': 'h',
               'bor': 'D'
               }
    for perplexity in np.arange(5, 76):
        tsne = TSNE(perplexity=perplexity,
                    random_state=0,
                    n_iter=10000,
                    n_iter_without_progress=5000,
                    init='pca',
                    learning_rate=0.1,
                    early_exaggeration=2)
        print('Fitting TSNE (perplexity = {})...'.format(perplexity))
        tsne_obj = tsne.fit_transform(X)
        print('Plotting...')
        directory = 'plots\\tsne\\{}\\'.format(dataset.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for attempt in range(6):
            try:
                tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                                        'Y': tsne_obj[:, 1],
                                        'Emotion': y})
                sns.scatterplot(x="X", y="Y",
                                hue="Emotion",
                                hue_order=hue_order,
                                legend='full',
                                data=tsne_df,
                                style='Emotion',
                                markers=markers,
                                s=30,
                                alpha=0.9,
                                palette=palette,
                                linewidth=0.3,
                                edgecolor='k'
                                )
                plt.title('{} T-SNE'.format(dataset.name))
                filename = directory + '{}_tsne_{}.png'.format(dataset.name, perplexity)
                plt.savefig(fname=filename)
                print('Saved {}'.format(filename))
                plt.close()
            except ValueError:
                raise
            break

if __name__ == '__main__':
    pass