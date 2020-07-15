from feature_extractor import *
from constants import *


class Dataset:
    def __init__(self, name, folder, language, label):
        self.name = name
        self.folder = folder
        self.language = language
        self.feature_folder = folder + 'features\\'
        self.feature_extractor = FeatureExtractor(folder, EXE_PATH, CONFIG_PATH, self.feature_folder)
        self.path_to_feature_file = self.feature_folder + '_features_with_labels.csv'
        if not os.path.exists(self.path_to_feature_file):
            self.feature_extractor.extract()
        self.features = FeatureFile(self.path_to_feature_file, label)
        self.label = label


class FeatureFile:
    def __init__(self, path, label):
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_xy(label)

    def _get_xy(self, label):
        df = self.contents
        X = df.drop(['name', 'label', 'frameTime', 'label_2', 'label_3'], axis=1).values
        if label == 'original':
            y = df[['label']].values.ravel()
        elif label == 'pos_neg_neu':
            y = df[['label_2']].values.ravel()
        elif label == 'negative_binary':
            y = df[['label_3']].values.ravel()
        else:
            raise ValueError('Unknown value for "label": should be either "original" or "pos_neg_neu"')
        return X, y


def create_english_dataset():
    datasets = [iemo_original, ravdess_original, cremad_original, savee_original, tess_original]
    big_dataset_features_list = []
    for dataset in datasets:
        big_dataset_features_list.append(dataset.features.contents)
    all_english = pd.concat(big_dataset_features_list, ignore_index=True)
    all_english.to_csv('datasets\\all_english\\features_with_labels.csv', index=False, sep=';')
    print('Done')


def relabel(dataset):
    print('Working with dataset {}'.format(dataset.name))
    features = dataset.features.contents
    print('Read file success')
    features['label_3'] = features['label_2']
    features.loc[(features['label_2'] == 'pos') | (features['label_2'] == 'neu'), 'label_3'] = 'not'
    print('Relabel success')
    features.to_csv(dataset.path_to_feature_file, index=False, sep=';')
    print('Write file success')


if __name__ == '__main__':
    pass

iemo_original = Dataset('Iemocap', IEMOCAP_FOLDER, 'English', label='original')
ravdess_original = Dataset('RAVDESS', RAVDESS_FOLDER, 'English', label='original')
cremad_original = Dataset('Crema-D', CREMAD_FOLDER, 'English', label='original')
savee_original = Dataset('SAVEE', SAVEE_FOLDER, 'English', label='original')
tess_original = Dataset('TESS', TESS_FOLDER, 'English', label='original')
emodb_original = Dataset('Emo-DB', EMODB_FOLDER, 'German', label='original')
all_english_six_original = Dataset('English-Assembly-Six', ASSEMBLY_SIX_FOLDER, 'English', label='original')
df = all_english_six_original.features.contents
df = df.loc[(df['label'] == 'ang') | (df['label'] == 'dis') | (df['label'] == 'fea') | (df['label'] == 'hap') | (df['label'] == 'neu') | (df['label'] == 'sad')]
all_english_six_original.features.contents = df
datasets_list_original = [savee_original,tess_original,
                          ravdess_original,cremad_original,
                          iemo_original, emodb_original,
                          all_english_six_original]

iemo_pos_neg_neu = Dataset('Iemocap-PosNegNeu', IEMOCAP_FOLDER, 'English', label='pos_neg_neu')
ravdess_pos_neg_neu = Dataset('RAVDESS-PosNegNeu', RAVDESS_FOLDER, 'English', label='pos_neg_neu')
cremad_pos_neg_neu = Dataset('Crema-D-PosNegNeu', CREMAD_FOLDER, 'English', label='pos_neg_neu')
savee_pos_neg_neu = Dataset('SAVEE-PosNegNeu', SAVEE_FOLDER, 'English', label='pos_neg_neu')
tess_pos_neg_neu = Dataset('TESS-PosNegNeu', TESS_FOLDER, 'English', label='pos_neg_neu')
emodb_pos_neg_neu = Dataset('Emo-DB-PosNegNeu', EMODB_FOLDER, 'German', label='pos_neg_neu')
all_english_six_pos_neg_neu = Dataset('English-Assembly-Six-PosNegNeu', ASSEMBLY_SIX_FOLDER, 'English',
                                      label='pos_neg_neu')
datasets_list_pos_neg_neu = [iemo_pos_neg_neu, ravdess_pos_neg_neu,
                             cremad_pos_neg_neu, savee_pos_neg_neu,
                             tess_pos_neg_neu, emodb_pos_neg_neu,
                             all_english_six_pos_neg_neu]

iemo_negative_binary = Dataset('Iemocap-NegativeBinary', IEMOCAP_FOLDER, 'English', label='negative_binary')
ravdess_negative_binary = Dataset('RAVDESS-NegativeBinary', RAVDESS_FOLDER, 'English', label='negative_binary')
cremad_negative_binary = Dataset('Crema-D-NegativeBinary', CREMAD_FOLDER, 'English', label='negative_binary')
savee_negative_binary = Dataset('SAVEE-NegativeBinary', SAVEE_FOLDER, 'English', label='negative_binary')
tess_negative_binary = Dataset('TESS-NegativeBinary', TESS_FOLDER, 'English', label='negative_binary')
emodb_negative_binary = Dataset('Emo-DB-NegativeBinary', EMODB_FOLDER, 'German', label='negative_binary')
all_english_six_negative_binary = Dataset('English-Assembly-Six-NegativeBinary', ASSEMBLY_SIX_FOLDER, 'English',
                                          label='negative_binary')
datasets_list_negative_binary = [emodb_negative_binary, ravdess_negative_binary,
                                 cremad_negative_binary, savee_negative_binary,
                                 tess_negative_binary, iemo_negative_binary,
                                 all_english_six_negative_binary]

if __name__ == '__main__':
    print(all_english_six_original.features.contents.groupby('label').count())