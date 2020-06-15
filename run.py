from feature_extractor import *
from constants import *


Extractor = FeatureExtractor(INPUT_FOLDER, EXE_PATH, CONFIG_PATH, OUTPUT_FOLDER)
Extractor.extract()
