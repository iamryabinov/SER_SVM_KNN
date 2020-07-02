from knn_multiclass import *
    

datasets_n = [(emodb, 14), (iemo, 21), (ravdess, 2), (cremad, 76), (savee, 5), (tess, 9), (all_english_six, 13)]
for dataset, n in datasets_n:
    knn_confusion_matrix(dataset, n)
