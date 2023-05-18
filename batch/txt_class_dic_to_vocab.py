import os

import pandas as pd
from pandarallel import pandarallel

from common.ClassFile import ClassFile
from preprocess.lib.TextPreprocess import TextPreprocess

os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LIBRARIES_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess.lib.NlpTool import NlpTool


class GbcNlpService:
    ENCODING = 'utf-8'
    DATA_DIR = rf''
    _LOAD_MODEL_ROOT = ''
    VOCAB_LIB = f'{_LOAD_MODEL_ROOT}.vocab'

    def __init__(self):
        self.label_dict = {}
        self.vocab = ''
        self.nlp_service = NlpTool()

    def _loader(self, path, d_list, c_list):
        doc_list = ClassFile.list_files_like(path, 'txt')
        print(f"Total documents: {len(doc_list)}")
        for doc in doc_list:
            try:
                page_text = ClassFile.get_text(doc, encoding=self.ENCODING)
                full_text = TextPreprocess.load_document(page_text)
            except Exception as _:
                continue
            print('.', end='')
            d_list.append(doc)
            c_list.append(full_text)
        print()

        return d_list, c_list

    def text_to_vocab(self, dataset):
        if not os.path.isdir(dataset):
            raise ValueError(f"Missing data directory at '{dataset}'")

        # build vocabulary
        try:
            self.vocab = self.nlp_service.load_vocabulary('', f"{self.VOCAB_LIB}")
            if self.vocab is None or not self.vocab:
                raise ValueError("No vocabulary was found")
        except Exception as _:
            raise ValueError("No vocabulary was found")

        print(f"Loading from {dataset}")
        doc_list, cont_list = self._loader(self.DATA_DIR, list(), list())
        df = pd.DataFrame(
            list(zip(doc_list, cont_list)), columns=['Document', 'Content'])
        print(f"\n{df.head()}")
        pandarallel.initialize(progress_bar=True, nb_workers=4)
        steam = False
        is_spanish = False
        token_filter = False
        min_vocab = 10
        vocab_dist = 3
        greater = 3
        df["Content"] = df["Content"].parallel_apply(
            lambda x: ' '.join(self.nlp_service.filter_word_list(
                x,
                stem=steam,
                is_spanish=is_spanish,
                vocab=self.vocab,
                greater=greater,
                token_filter=False,
                vocab_dist=vocab_dist)))
        print(f"\nCleaned: {df.sample(10)}")
        df.dropna(
            axis=0,
            how='any',
            thresh=None,
            subset=None,
            inplace=True
        )
        print(f"\nDropped: {df.sample(10)}")

        print(f"Saving cleaned files to {dataset}")
        for ind in df.index:
            ClassFile.to_txtfile(
                data=df['Content'][ind],
                file_=df['Document'][ind].replace('.txt', '.vocab'),
                mode='w+',
                encoding=self.ENCODING)


def main():
    print(f"\nPre-process data from {GbcNlpService.DATA_DIR}...")
    text_service = GbcNlpService()
    text_service.text_to_vocab(GbcNlpService.DATA_DIR)


if __name__ == '__main__':
    main()
