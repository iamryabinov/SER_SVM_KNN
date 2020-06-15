import os


class FeatureExtractor:
    def __init__(self, input_folder, exe_path, config_path, output_folder):
        self.input_folder = input_folder
        self.exe_path = exe_path
        self.config_path = config_path
        self.output_folder = output_folder

    def construct_call(self, input_file_name):
        instance_name = input_file_name.split('.')[0]
        output_file_name = instance_name + '.csv'
        config_options = ' -C ' + self.config_path
        input_options = ' -I ' + self.input_folder + input_file_name
        output_options = ' -lldcsvoutput ' + self.output_folder + output_file_name
        instance_options = ' -instname ' + instance_name
        misc = ''
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
