# from dataset import *
# from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import matplotlib as mpl



def plot_tsne(dataset):
    print('\nWORKING WITH {}'.format(dataset.name))
    dataset.name = dataset.name.split('_')[0].lower()
    X, y = dataset.features.X, dataset.features.y.ravel()
    X = StandardScaler().fit_transform(X)
    if len(y) > 1000:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=1000)
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
        directory = 'plots\\tsne\\{}_{}\\'.format(dataset.name, dataset.label)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for attempt in range(6):
            try:
                tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                                        'Y': tsne_obj[:, 1],
                                        'Emotion': y})
                sns.scatterplot(x="X", y="Y",
                                hue="Emotion",

                                legend='full',
                                data=tsne_df,
                                style='Emotion',

                                s=30,
                                alpha=0.9,

                                linewidth=0.3,
                                edgecolor='k'
                                )   #hue_order = hue_order, markers=markers, palette=palette,
                plt.title('{} T-SNE'.format(dataset.name))
                filename = directory + '{}_tsne_{}.png'.format(dataset.name, perplexity)
                plt.savefig(fname=filename)
                print('Saved {}'.format(filename))
                plt.close()
            except ValueError:
                raise
            break


def create_tsne_file():
    tsne_dfs = []
    dataset_perplexity = {
        iemo_original: 70,
        cremad_original: 22,
        all_english_six_original: 71,
        emodb_original: 37,
        ravdess_original: 41,
        savee_original: 21,
        tess_original: 37
    }
    for dataset in datasets_list_original:
        print('\nWORKING WITH {}'.format(dataset.name))
        dataset.name = dataset.name.split('_')[0].lower()
        X, y = dataset.features.X, dataset.features.y.ravel()
        X = StandardScaler().fit_transform(X)
        if len(y) > 1001:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=1000)
            X = X_test
            y = y_test
            print('Fetched data')
        perplexity = dataset_perplexity[dataset]
        tsne = TSNE(perplexity=perplexity,
                    random_state=0,
                    n_iter=5000,
                    n_iter_without_progress=2000,
                    init='pca',
                    learning_rate=0.1,
                    early_exaggeration=2)
        print('Fitting TSNE (perplexity = {})...'.format(perplexity))
        tsne_obj = tsne.fit_transform(X)
        print('Creating Dataframe...')
        tsne_df = pd.DataFrame({'Dataset': dataset.name,
                                'X': tsne_obj[:, 0],
                                'Y': tsne_obj[:, 1],
                                'Label': y})
        tsne_dfs.append(tsne_df)
        print('Appended')
    print('Concatenating...')
    df = pd.concat(tsne_dfs, ignore_index=True)
    df.to_csv('tsne_results.csv', sep=';', index=False)

def visualize_tsne():
    mpl.use('Qt5Agg')
    data = pd.read_csv('tsne_results.csv', delimiter=';')
    data = data.loc[data['Label type'] == 'Binary']
    data = data.loc[data['Dataset'] != 'English Assembly Six']
    col_order = ['Crema-D', 'Emo-DB', 'IEMOCAP', 'RAVDESS', 'SAVEE', 'TESS']
    hue_order = ['ang', 'hap', 'neu', 'sad', 'dis', 'fea', 'sur', 'cal', 'exc', 'fru', 'bor', 'neg', 'rest']
    sns.set(font="Arial", style='white', context='paper', font_scale=1.5)
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
               'bor': '#A78A00',
               'neg': '#FF0000',
               'rest': '#23FF00'
               }
    markers = {
        'neg': 'x',
        'rest': '+'
    }
    g = sns.relplot(x="X", y="Y",
                    hue="Label", hue_order=hue_order, palette=palette,
                    col='Dataset', col_wrap=3, col_order=col_order,
                    legend='full', height=4, aspect=1,
                    kind="scatter", data=data, facet_kws={'sharex': False, 'sharey': False},
                    s=15, linewidth=0, 
                    )
    for ax, title in zip(g.axes.flat, ['а', 'б', 'в', 'г', 'д', 'е']):
        ax.set_title('')
        ax.set_ylabel('')
        ax.set_xlabel(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.title.set_position([0.3, -0.01])
    sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    plt.show()



if __name__ == '__main__':
    visualize_tsne()

