from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print('RAVDESS')
df = pd.read_csv('datasets\\ravdess\\features\\_features_with_labels.csv', delimiter=';')
X = df.drop(['name', 'label', 'frameTime', 'label_2'], axis=1).values
y = df[['label_2']].values.ravel()
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
scores = {
    'n': [],
    'test': []
}
for n in np.arange(1, 41):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    scores['n'].append(n)
    scores['test'].append(test_score)
    print('K-NN with {} neighbors, train score: {}, test score: {}'.format(n, train_score, test_score))
test_max = max(scores['test'])
test_max_index = index(max(scores['test']))
n_max = scores['n'][test_max_index]    
print(f'max test score {test_max} with {n_max} neighbors')
knn = KNeighborsClassifier(n_neighbors=n_max)
knn.fit(X_train, y_train)
disp = plot_confusion_matrix(
    knn, X_test, y_test,
    normalize='true', xticks_rotation=30, cmap=plt.cm.Oranges,
    )
plt.show()
