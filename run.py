from knn_multiclass import *
    
iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
emodb = Dataset('Emo-DB', EMODB_FOLDER, 'German')
ravdess = Dataset('Ravdess', RAVDESS_FOLDER, 'English')
cremad = Dataset('Crema-D', CREMAD_FOLDER, 'English')
savee = Dataset('SAVEE', SAVEE_FOLDER, 'English')
tess = Dataset('TESS', TESS_FOLDER, 'English')
datasets_list = [emodb, iemo, ravdess, cremad, savee, tess]

knn_multi_classification(datasets_list)