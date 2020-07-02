from dataset import *
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from knn_multiclass import standardize


datasets = [emodb, cremad, all_english_six, iemo, ravdess, savee, tess]
for dataset in datasets:
    hue_order = ['ang', 'hap', 'sad', 'neu', 'dis', 'fea', 'sur', 'cal', 'exc', 'fru', 'bor']
    palette = ['#FF0000', '#23FF00', '#3D00FF', '#5D5D5D', '#A400FF', '#00FFFF', '#00A700',
               '#A78A00', '#FFD300', '#FF7400', '#A78A00']
    markers = ['o', 's', 'P', 'X', '^', 'v', '*', 'p', 'D', 'h', 'D']
    if dataset == cremad:
        hue_order, palette, markers = hue_order[:6], palette[:6], markers[:6]
    if dataset == emodb:
        hue_order, palette, markers = hue_order[:6] + [hue_order[-1]], \
                                      palette[:6] + [palette[-1]], markers[:6] + [markers[-1]]
    if dataset == all_english_six:
        hue_order, palette, markers = hue_order[:6], palette[:6], markers[:6]
    if dataset == iemo:
        hue_order, palette, markers = hue_order[:4] + hue_order[8:10], \
                                      palette[:4] + palette[8:10], markers[:4] + markers[8:10]
    if dataset == ravdess:
        hue_order, palette, markers = hue_order[:8], palette[:8], markers[:8]
    if dataset == savee:
        hue_order, palette, markers = hue_order[:7], palette[:7], markers[:7]
    if dataset == tess:
        hue_order, palette, markers = hue_order[:7], palette[:7], markers[:7]
    print('\nWORKING WITH {}'.format(dataset.name))
    standardize([dataset])
    dataset.name = dataset.name.split('_')[0].lower()
    X, y = dataset.features.X, dataset.features.y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)
    if len(y) > 2500:
        X = X_test
        y = y_test
        print('Fetched data')
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
