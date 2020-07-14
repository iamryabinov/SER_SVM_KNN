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