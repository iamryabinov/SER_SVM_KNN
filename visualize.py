import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd


sns.set(style="ticks")
results = pd.read_csv('classifiers\\knn\\knn_summarized_results_final.csv', delimiter=';')
g = sns.relplot(x="n_neighbors", y="score",
                hue="subset", style="subset", col="dataset", col_wrap=3,
                legend='brief', height=20, aspect=.5,
                kind="line", data=results)
plt.show()