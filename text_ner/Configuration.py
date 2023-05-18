import os

import yaml


class Configuration:
    def __init__(self, conf_route="config.yml", working_path=""):
        try:
            self.dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(self.dir_path, conf_route), 'r') as yml_file:
                cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
        except Exception as e:
            print('Configuration::__init__::{0}::{1}'.format(conf_route, str(e.args)))
            exit(1)

        development = cfg['development']

        # working directories
        directories = development['directories']
        self.lang = directories['lang']
        self.working_path = working_path

        # spacy models
        spacy = development['spacy']
        self.base_model = spacy['base_model']
        self.name_model = spacy['name_model']

        # transformer models
        transformer = development['transformer']
        self.berto_model = transformer['berto_model']
        self.ruperta_model = transformer['ruperta_model']

        # system separator
        self.sep = os.path.sep
