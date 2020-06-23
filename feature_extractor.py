import os
import pandas as pd

class FeatureExtractor:
    def __init__(self, input_folder, exe_path, config_path, output_folder):
        self.input_folder = input_folder
        self.exe_path = exe_path
        self.config_path = config_path
        self.output_folder = output_folder

    def construct_call(self, input_file_name):
        instance_name = input_file_name.split('.')[0]
        config_options = ' -C ' + self.config_path
        input_options = ' -I ' + self.input_folder + input_file_name
        output_options = ' -csvoutput ' + self.output_folder + 'features.csv'
        instance_options = ' -instname ' + instance_name
        misc = ' -loglevel 1 -nologfile'
        opensmile_call = self.exe_path + config_options + input_options + output_options + instance_options + misc
        return opensmile_call

    def extract(self):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        input_files = os.listdir(self.input_folder)
        for input_file_name in input_files:
            opensmile_call = self.construct_call(input_file_name)
            os.system(opensmile_call)
            print(f'Extracted from {input_file_name} successfully')
        features = pd.read_csv(self.output_folder + 'features.csv', delimiter=';')
        features['label'] = ''
        cols = list(features.columns.values)
        features = features[cols[0:1] + [cols[-1]] + cols[1:-1]]
        features.to_csv(self.output_folder + 'features_with_labels.csv', index=False, sep=';')
