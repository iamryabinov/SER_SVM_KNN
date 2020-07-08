from feature_extractor import *
from constants import *


class Dataset:
    def __init__(self, name, folder, language, label):
        self.name = name
        self.folder = folder
        self.language = language
        self.feature_folder = folder + 'features\\'
        self.feature_extractor = FeatureExtractor(folder, EXE_PATH, CONFIG_PATH, self.feature_folder)
        path_to_feature_file = self.feature_folder + '_features_with_labels.csv'
        if not os.path.exists(path_to_feature_file):
            self.feature_extractor.extract()
        self.features = FeatureFile(path_to_feature_file, label)
        self.label = label


class FeatureFile:
    def __init__(self, path, label):
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_xy(label)

    def _get_xy(self, label):
        df = self.contents
        X = df.drop(['name', 'label', 'frameTime', 'label_2'], axis=1).values
        if label == 'original':
            y = df[['label']].values.ravel()
        elif label == 'pos_neg_neu':
            y = df[['label_2']].values.ravel()
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


if __name__ == '__main__':
    pass

iemo_original = Dataset('Iemocap', IEMOCAP_FOLDER, 'English', label='original')
ravdess_original = Dataset('Ravdess', RAVDESS_FOLDER, 'English', label='original')
cremad_original = Dataset('Crema-D', CREMAD_FOLDER, 'English', label='original')
savee_original = Dataset('SAVEE', SAVEE_FOLDER, 'English', label='original')
tess_original = Dataset('TESS', TESS_FOLDER, 'English', label='original')
emodb_original = Dataset('Emo-DB', EMODB_FOLDER, 'German', label='original')
all_english_six_original = Dataset('English-Assembly-Six', ASSEMBLY_SIX_FOLDER, 'English', label='original')
datasets_list_original = [iemo_original, ravdess_original,
                          cremad_original, savee_original,
                          tess_original, emodb_original,
                          all_english_six_original]

iemo_pos_neg_neu = Dataset('Iemocap-PosNegNeu', IEMOCAP_FOLDER, 'English', label='pos_neg_neu')
ravdess_pos_neg_neu = Dataset('Ravdess-PosNegNeu', RAVDESS_FOLDER, 'English', label='pos_neg_neu')
cremad_pos_neg_neu = Dataset('Crema-DPosNegNeu', CREMAD_FOLDER, 'English', label='pos_neg_neu')
savee_pos_neg_neu = Dataset('SAVEE-PosNegNeu', SAVEE_FOLDER, 'English', label='pos_neg_neu')
tess_pos_neg_neu = Dataset('TESS-PosNegNeu', TESS_FOLDER, 'English', label='pos_neg_neu')
emodb_pos_neg_neu = Dataset('Emo-DB-PosNegNeu', EMODB_FOLDER, 'German', label='pos_neg_neu')
all_english_six_pos_neg_neu = Dataset('English-Assembly-Six-PosNegNeu', ASSEMBLY_SIX_FOLDER, 'English', label='pos_neg_neu')
datasets_list_pos_neg_neu = [iemo_pos_neg_neu, ravdess_pos_neg_neu,
                             cremad_pos_neg_neu, savee_pos_neg_neu,
                             tess_pos_neg_neu, emodb_pos_neg_neu,
                             all_english_six_pos_neg_neu]
