from dataset import *
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import numpy as np

iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
X = iemo.features.X
y = iemo.features.y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
plot_confusion_matrix(knn, X_test, y_test, normalize='true')
plt.show()






