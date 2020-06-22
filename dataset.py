import pandas as pd


class Dataset:
    def __init__(self, folder, language):
        self.folder = folder
        self.language = language
        self.features = FeatureFile(folder + 'features\\features_with_labels.csv')


class FeatureFile:
    def __init__(self, path):
        self.name = path.split('//')[-1]
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_xy()

    def _get_xy(self):
        df = self.contents
        X = df.drop(['name', 'label', 'frameTime'], axis=1).values
        y = df[['label']].values
        return(X, y)


if __name__ == '__main__':
    Iemo = Dataset('datasets\\iemocap_audios\\', 'English')
    print(Iemo.folder)
    print(Iemo.language)
    print(Iemo.features.name)
    print(Iemo.features.contents.head(3))
    print(Iemo.features.X, Iemo.features.y)