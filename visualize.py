import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl


df_knn = pd.read_csv('classifiers\\knn\\knn_summarized_results_final.csv', delimiter=';')

def plot_multilabel_results():
    sns.set(font="Arial", style='ticks', palette="PuBuGn_d", context='notebook', font_scale=1.25,
        rc={
                        'axes.grid': True, 
                        'grid.linestyle': '--',  
                        'xtick.direction': 'out', 
                        'ytick.direction': 'out',
                        "lines.linewidth": 1.5,
                        "grid.linewidth": 0.6,

        })
    df_knn = df_knn.loc[df_knn['Dataset'] != 'English-Assembly-Six']
    df_knn = df_knn.loc[(df_knn['Labeling'] == 'Descrete basic emotions') & (df_knn['Subset'] == 'Test')]
    dataset_order = np.sort(df_knn.Dataset.unique()).tolist()
    g = sns.relplot(x="Neighbors", y="Model score",
                            hue="Preprocessing", hue_order=['Standardization', 'None', 'Normalization'],
                            style='Preprocessing', style_order=['Standardization', 'None', 'Normalization'],
                            col="Dataset", col_wrap=3,
                            legend='full', height=3, aspect=1,
                            kind="line", data=df_knn, facet_kws={'sharex': True, 'sharey': True, 'despine': False}
                        )
    g.set(xlim=(0, 76), ylim=(0, 1))
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        # plt.scatter(max_test_score_n, max_test_score,
        #             label='Max test score {:.3f} with {:.0f} neighbors'.format(max_test_score, max_test_score_n),
        #             c='red', marker=7, s=500, zorder=10)
    sns.despine()
    plt.show() 


if __name__ == '__main__':
    plot_multilabel_results()