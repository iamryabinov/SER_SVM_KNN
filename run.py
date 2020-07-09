import pandas as pd
from dataset import *

def relabel(dataset):
    print('Working with dataset {}'.format(dataset.name))
    features = dataset.features.contents
    print('Read file success')
    features['label_3'] = features['label_2']
    features.loc[(features['label_2'] == 'pos') | (features['label_2'] == 'neu'), 'label_3'] = 'not'
    print('Relabel success')
    features.to_csv(dataset.path_to_feature_file, index=False, sep=';')
    print('Write file success')

for dataset in datasets_list_pos_neg_neu:
    relabel(dataset)
print('Done!')