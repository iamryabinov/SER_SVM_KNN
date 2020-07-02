from feature_extractor import *
from constants import *

class Dataset:
    def __init__(self, name, folder, language):
        self.name = name
        self.folder = folder
        self.language = language
        self.feature_folder = folder + 'features\\'
        self.feature_extractor = FeatureExtractor(folder, EXE_PATH, CONFIG_PATH, self.feature_folder)
        path_to_feature_file = self.feature_folder + 'features_with_labels.csv'
        if not os.path.exists(path_to_feature_file):
            self.feature_extractor.extract()
        self.features = FeatureFile(path_to_feature_file)


class FeatureFile:
    def __init__(self, path):
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_xy()

    def _get_xy(self):
        df = self.contents
        X = df.drop(['name', 'label', 'frameTime'], axis=1).values
        y = df[['label']].values
        return(X, y)

def create_english_dataset():
    iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
    ravdess = Dataset('Ravdess', RAVDESS_FOLDER, 'English')
    cremad = Dataset('Crema-D', CREMAD_FOLDER, 'English')
    savee = Dataset('SAVEE', SAVEE_FOLDER, 'English')
    tess = Dataset('TESS', TESS_FOLDER, 'English')
    datasets = [iemo, ravdess, cremad, savee, tess]
    big_dataset_features_list = []
    for dataset in datasets:
        big_dataset_features_list.append(dataset.features.contents)
    all_english = pd.concat(big_dataset_features_list, ignore_index=True)
    all_english.to_csv('datasets\\all_english\\features_with_labels.csv', index=False, sep=';')
    print('Done')

if __name__ == '__main__':
    create_english_dataset()

iemo = Dataset('Iemocap', IEMOCAP_FOLDER, 'English')
ravdess = Dataset('Ravdess', RAVDESS_FOLDER, 'English')
cremad = Dataset('Crema-D', CREMAD_FOLDER, 'English')
savee = Dataset('SAVEE', SAVEE_FOLDER, 'English')
tess = Dataset('TESS', TESS_FOLDER, 'English')
emodb = Dataset('Emo-DB', EMODB_FOLDER, 'German')
all_english_six = Dataset('English-Assembly-Six', ASSEMBLY_SIX_FOLDER, 'English' )