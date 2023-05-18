import os

from pandas import DataFrame

from common.ClassFile import ClassFile
from preprocess.lib.TextPreprocess import TextPreprocess


class NlpTool:
    ENCODING = 'utf-8'

    def __init__(self):
        self.pre_process_service = TextPreprocess()

    def filter_word_list(self,
                         content,
                         vocab=None,
                         vocab_path=None,
                         stem=False,
                         is_spanish=True,
                         greater=3,
                         token_filter=False,
                         vocab_dist=3) -> list:
        if vocab_path is not None:
            token_list = self.pre_process_service.process_to_class(
                doc=content,
                vocab_path=vocab_path,
                stem=stem,
                is_spanish=is_spanish,
                greater=greater,
                token_filter=token_filter,
                vocab_dist=vocab_dist
            )
        else:
            token_list = self.pre_process_service.process_to_class(
                doc=content,
                vocab=vocab,
                stem=stem,
                is_spanish=is_spanish,
                greater=greater,
                token_filter=token_filter,
                vocab_dist=vocab_dist
            )

        return token_list

    def pre_process(self,
                    content,
                    is_spanish=True,
                    greater=3) -> list:
        token_list = self.pre_process_service.process_to_clean(
            doc=content
        )

        return token_list

    def load_vocabulary(self, vocab_path, vocab_load_name):
        path_to_vocab = os.path.join(vocab_path, vocab_load_name)
        dic_list = self.pre_process_service.load_vocabulary(path_to_vocab)
        if not dic_list:
            print(f"No vocabulary at '{path_to_vocab}'")

        return dic_list

    def generate_vocabulary(self,
                            data: object,
                            vocab_file_name: str = '',
                            stem: bool = False,
                            min_vocab: int = 2,
                            is_spanish: bool = True,
                            greater: int = 3):
        if isinstance(data, str):
            print(f"Create new vocabulary {vocab_file_name} from TXT files")
            data_list = ClassFile.list_files_ext(data, "txt")
            token_list = self.pre_process_service.process_vocabulary_from_file(
                data_list, stem=stem, min_vocab=min_vocab, is_spanish=is_spanish, greater=greater)
        elif isinstance(data, DataFrame):
            print(f"Create new vocabulary {vocab_file_name} from panda DataFrame")
            token_list = self.pre_process_service.process_vocabulary_from_doc(
                data.Content.values, stem=stem, min_vocab=min_vocab, is_spanish=is_spanish, greater=greater)
        else:
            token_list = None

        if vocab_file_name and token_list:
            print(f"Vocabulary saved to {vocab_file_name}")
            if os.path.isfile(vocab_file_name):
                os.remove(vocab_file_name)
            with open(vocab_file_name, "a+", encoding=self.ENCODING) as output:
                for token in token_list:
                    output.write(f"{token}\n")

        return token_list
