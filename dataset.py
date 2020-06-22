import os
import pandas as pd

class Dataset:
    def __init__(self, folder, language):
        self.folder = folder
        self.language = language
        self.wawfiles = [inst for inst in os.listdir(folder) if inst.endswith('.waw')]
        self.features_folder = folder + 'features\\' 



class FeatureFile:
    def __init__(self, path):
        self.name = path.split('//')[-1]
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_Xy()
    
    def _get_Xy(self):
        x, y = [0, 0]
        return(x, y)
