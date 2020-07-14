import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Qt5Agg')
sns.set_style("white")
sns.set_context("paper")
results = pd.read_csv('classifiers\\knn\\knn_summarized_results_final.csv', delimiter=';')
g = sns.relplot(x="n_neighbors", y="score",
                hue="subset", style="subset", col="dataset", col_wrap=3,
                legend='brief', height=3, aspect=1,
                kind="line", data=results, palette="PuBuGn_d")
plt.show()

sns.set_style("ticks", {'axes.grid': True, 
                        'grid.linestyle': '--',  
                        'xtick.direction': 'out', 'ytick.direction': 'out',
                        })
sns.set_context("paper")

df = pd.read_csv('classifiers\\knn\\knn_summarized_results_final.csv', delimiter=';')
df = df.loc[df['Dataset'] != 'English-Assembly-Six']
df = df.loc[(df['Labeling'] == 'Descrete basic emotions') & (df['Subset'] == 'Test')]



g = sns.relplot(x="Neighbors", y="Model score",
                hue="Preprocessing", style='Preprocessing', style_order=['Standardization', 'None', 'Normalization'],
                col="Dataset", col_wrap=3,
                legend='brief', height=3, aspect=1,
                kind="line", data=df, facet_kws={'sharex': False, 'sharey': True, 'despine': False})
g.despine()
for ax in g.axes.flat:
    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
g.set(xlim=(0, 60), ylim=(0, 1))
plt.show()
