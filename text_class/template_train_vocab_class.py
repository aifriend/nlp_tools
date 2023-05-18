import os

import pandas as pd

os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LIBRARIES_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess.lib.NlpTool import NlpTool


class GbcNlpService:
    SOURCE = './'
    DF_CONTENT = 'Content'
    DF_CATEGORY = 'Category'
    _LOAD_MODEL_ROOT = './'
    DATA_DIR = rf''
    VOCAB_LIB = f'{_LOAD_MODEL_ROOT}.model.vocab'

    # BETO Bert spanish pre-trained
    BERT_MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'

    def __init__(self):
        self.df = None
        self.label_dict = {}
        self.vocab = ''
        self.dataset_train = None
        self.dataset_val = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.nlp_service = NlpTool()

    def load_data_from_text(self, dataset):
        if not os.path.isdir(dataset):
            raise ValueError(f"Missing data directory at '{dataset}'")

        # build vocabulary
        steam = False
        is_spanish = False
        greater = 3
        min_vocab = 2
        self.vocab = self.nlp_service.generate_vocabulary(
            dataset, vocab_file_name=self.VOCAB_LIB,
            stem=steam, min_vocab=min_vocab, is_spanish=is_spanish, greater=greater)
        if self.vocab is None or not self.vocab:
            raise ValueError("No vocabulary was found")

    def load_data_from_csv(self, dataset):
        print(f"Loading from {dataset}...")
        self.df = pd.read_csv(dataset, skip_blank_lines=True, delimiter=",")

        possible_labels = self.df.Category.unique()
        self.label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            self.label_dict[possible_label] = index
        print(f"\nClasses: {self.label_dict}")

        # add label_text by numbers
        self.df['label'] = self.df.Category.replace(self.label_dict)
        print(f"\n{self.df.head()}")

        # vocabulary pre-process
        if os.path.isfile(self.VOCAB_LIB):
            os.remove(self.VOCAB_LIB)
        steam = False
        is_spanish = False
        greater = 3
        min_vocab = 1
        self.vocab = self.nlp_service.generate_vocabulary(
            self.df, vocab_file_name=self.VOCAB_LIB,
            stem=steam, min_vocab=min_vocab, is_spanish=is_spanish, greater=greater)
        if self.vocab is None or not self.vocab:
            print("No vocabulary was found")
        print(f"\n{self.vocab}")


def main():
    text_service = GbcNlpService()

    # print(f"\nPre-process data from {GbcNlpService.DATA_DIR}...")
    # text_service.load_data_from_text(GbcNlpService.DATA_DIR)

    print(f"\nPre-process data from{GbcNlpService.SOURCE}...")
    text_service.load_data_from_csv(GbcNlpService.SOURCE)


if __name__ == '__main__':
    main()
