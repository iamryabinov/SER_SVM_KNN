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
