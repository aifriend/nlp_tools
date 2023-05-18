import base64
import os
import re
import string
from collections import Counter
from multiprocessing import Pool

import unicodedata
from Levenshtein import distance
from nltk import data
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

from lib.ClassFile import ClassFile
from lib.SpacyModel import SpacyModel


class TextPreprocess:
    ENCODING = "utf-8"

    def __init__(self):
        try:
            self.spell = SpellChecker(
                language='es', distance=2
            )
            self.spanish_tokenizer = self.load_tokenizer(
                tokenizer_path='tokenizers/punkt/spanish.pickle'
            )
            self.stop_words = set(stopwords.words('spanish'))
            self.doc_nlp = SpacyModel.getInstance().getModel()
        except Exception as _:
            raise IOError('TextPreprocess::Initialization error')

    def process_to_class(self,
                         doc: str,
                         vocab: list = None,
                         vocab_path: str = None,
                         stem: bool = False,
                         greater: int = 3,
                         is_spanish: bool = True,
                         vocab_dist: int = 3) -> list:
        # clean punctuation marks and non-words text
        encoded_doc = doc.encode(self.ENCODING, "ignore")
        decoded_doc = encoded_doc.decode(self.ENCODING)

        # split into tokens by white space
        content_list = list()
        if self.spanish_tokenizer is not None:
            sentences = self.spanish_tokenizer.tokenize(decoded_doc)
            for s in sentences:
                to_tokenizer = self.clean_text(s)
                to_tokenizer = self.simplify(to_tokenizer)
                content_list.extend([s for s in word_tokenize(to_tokenizer)])
        else:
            content_list = decoded_doc.split()

        # lower case
        content_list = self.lower_case(content_list)

        # greater than
        if greater:
            content_list = self.greater_than(content_list, greater)

        # token cleaner
        content_list = self.filter_alpha_and_stop_word(content_list)

        # spacy token class filter
        content_list = self.spacy_token_filter(content_list)

        # spanish language
        if is_spanish:
            content_list = self.is_spanish(content_list)

        # stemmer vocab shorter
        if stem:
            content_list = self.stemmer(content_list)

        # filter token in model vocabulary list
        if vocab or vocab_path is not None:
            content_list = self.model_vocab_filter(
                token_list=content_list,
                vocab=vocab,
                vocab_path=vocab_path,
                vocab_dist=vocab_dist)

        return content_list

    def process_to_ner(self,
                       doc: str,
                       is_spanish: bool = True):
        # clean punctuation marks and non-words text
        content = self.load_document(doc)

        # split into tokens by white space
        content_list = list()
        if self.spanish_tokenizer is not None:
            sentences = self.spanish_tokenizer.tokenize(content)
            for s in sentences:
                to_tokenizer = self.clean_text(s)
                to_tokenizer = self.simplify(to_tokenizer)
                content_list.extend([s for s in word_tokenize(to_tokenizer)])
        else:
            content_list = content.split()

        # spanish language
        if is_spanish:
            content_list = self.is_spanish(content_list)

        return content_list

    @staticmethod
    def filter_token_min_occurrence(dic_list: list, min_occur=2):
        if not dic_list:
            return dic_list

        counter = Counter(dic_list)
        token_list = [(k, c) for k, c in counter.items() if c >= min_occur]
        sorted_token_list = list(map(
            lambda x: x[0], sorted(token_list, key=lambda x: x[1], reverse=True)))

        return sorted_token_list

    def is_spanish(self, work_list: list) -> list:
        if not work_list:
            return work_list

        # find those words that may be misspelled
        misspelled = self.spell.unknown(work_list)
        spelled = self.spell.known(work_list)

        ratio = len(spelled) / len(misspelled) if misspelled else 1.0
        if ratio <= 0.1:
            return list()

        # print(f"This words are no Spanish: {';'.join(misspelled)}")

        return [w for w in work_list
                if w in spelled or w.isalnum()]

    @staticmethod
    def load_tokenizer(tokenizer_path='tokenizers/punkt/spanish.pickle'):
        try:
            _tokenizer = data.load(
                resource_url=tokenizer_path
            )
        except Exception as _:
            _tokenizer = None

        return _tokenizer

    def spacy_token_filter(self, content_list: list):
        name_list = list()
        no_name_list = list()
        for content in content_list:
            token = next(iter(self.doc_nlp(content)))
            if (
                    token.pos_ == r'PROPN' or
                    token.pos_ == r'NOUN' or
                    token.pos_ == r'ADJ' or
                    token.pos_ == r'ADV' or
                    token.pos_ == r'AUX' or
                    token.pos_ == r'INTJ' or
                    token.pos_ == r'VERB'):
                name_list.append(token.text)
            else:
                no_name_list.append(token.text)

        # ClassFile.to_txtfile("\n".join(list(set(content_list) ^ set(name_list))),
        #                      file_='xor.vocab',
        #                      mode='a+',
        #                      encoding=self.ENCODING)

        return name_list

    @staticmethod
    def lower_case(token_list: list):
        filter_token_list = [w.lower() for w in token_list]
        return filter_token_list

    def model_vocab_filter(self, token_list: list, vocab: list, vocab_path: str, vocab_dist: int = 3):
        if not vocab and not vocab_path:
            raise ValueError('No vocabulary was loaded')

        # build vocabulary
        model_token_list = token_list.copy()
        if not vocab:
            vocab_list = self.load_vocabulary(vocab_path)
            if vocab_list:
                model_token_list = [w for w in token_list
                                    if any(filter(lambda x: distance(w.lower(), x) <= vocab_dist, vocab_list))]
            else:
                print("No vocabulary was filtered")
        else:
            model_token_list = [w for w in token_list
                                if any(filter(lambda x: distance(w.lower(), x) <= vocab_dist, vocab))]

        return model_token_list

    @staticmethod
    def greater_than(token_list: list, greater_than: int):
        lower_token_list = [w for w in token_list if len(w.lower()) >= greater_than]
        return lower_token_list

    @staticmethod
    def stemmer(content_list: list) -> list:
        spanish_stemmer = SnowballStemmer('spanish')
        token_list = [spanish_stemmer.stem(w) for w in content_list]
        return token_list

    @staticmethod
    def clean_text(text: str):
        if text is None:
            return ''

        new_ = re.sub(r'\n', ' ', text)
        new_ = re.sub(r'\n\n', r'\n', new_)
        new_ = re.sub(r'\\n', ' ', new_)
        new_ = re.sub(r'\t', ' ', new_)

        # page number
        new_ = re.sub(r'(\[\[\[.{1,3}\]\]\])', '', new_)
        new_ = re.sub(r'\- F.?o.?l.?i.?o \d{1,2} \-', '', new_)
        new_ = re.sub(r"F.?o.?l.?i.?o.{0,2}-.{0,2}[0-9]{1,2}.{0,2}-", '', new_)

        new_ = re.sub(r'(\d{5})-(\W*)(\w{3,20}?)\W', r'\1 \2', new_)

        new_ = re.sub(r'--', r'-', new_)
        new_ = re.sub(r'=', ' ', new_)
        new_ = re.sub(r':', ' ', new_)
        new_ = re.sub(r'\( ', r'(', new_)
        new_ = re.sub(r' \)', r')', new_)
        new_ = re.sub(r'"', ' ', new_)
        new_ = re.sub(r'_', ' ', new_)
        new_ = re.sub(r'\'', ' ', new_)
        new_ = re.sub(r'/', ' ', new_)

        new_ = re.sub(r'(\s[a-z0-9]{1,3})-([a-z0-9]{1,3}\s)', r'\1 \2', new_)
        new_ = re.sub(r'\s{2,1000}', " ", new_)
        new_ = re.sub(r"(\w{2})-\s(\w{2})", r'\1\2', new_)  # join new line words

        return " ".join(new_.split()).strip()

    @staticmethod
    def flat_text(text: str):
        if text is None:
            return ''

        new_ = re.sub(r'\n', ' ', text)
        new_ = re.sub(r'\n\n', r'\n', new_)
        new_ = re.sub(r'\\n', ' ', new_)
        new_ = re.sub(r'\t', ' ', new_)
        new_ = re.sub(r'\s{2,1000}', ' ', new_)

        # page number
        new_ = re.sub(r'(\[\[\[.{1,3}\]\]\])', '', new_)
        new_ = re.sub(r'\- F.?o.?l.?i.?o \d{1,2} \-', '', new_)
        new_ = re.sub(r"F.?o.?l.?i.?o.{0,2}-.{0,2}[0-9]{1,2}.{0,2}-", '', new_)

        return new_.strip()

    @staticmethod
    def simplify(text: str):
        clean_text = text
        try:
            chars = [c for c in unicodedata.normalize('NFD', text)
                     if c not in string.punctuation]
            if chars:
                clean_text = unicodedata.normalize('NFC', ''.join(chars))
        except NameError:
            pass

        return clean_text

    def filter_alpha_and_stop_word(self, token_list: list) -> list:
        # remove not alphabetic tokens
        filter_token_list = [word for word in token_list
                             if word.isalpha() or word.isalnum()]

        # filter out stop words
        filter_token_list = [word for word in filter_token_list
                             if word not in self.stop_words]

        return filter_token_list

    @staticmethod
    def load_vocabulary(vocab_path):
        _vocab = list()
        try:
            with open(vocab_path, "r", encoding=TextPreprocess.ENCODING) as f_vocab:
                _content = f_vocab.readlines()
                _vocab = list(map(lambda x: x[:-1], _content))
            print(f"Load vocabulary from {vocab_path} with UTF-8 encoder...")
        except Exception as _:
            try:
                if os.path.isfile(vocab_path):
                    with open(vocab_path, "r") as f_vocab:
                        _content = f_vocab.readlines()
                        _vocab = list(map(lambda x: x[:-1], _content))
                    print(f"Load vocabulary from {vocab_path} with default encoder...")
            except Exception as _:
                try:
                    if os.path.isfile(vocab_path):
                        with open(vocab_path, "r") as f_vocab:
                            _vocab = f_vocab.readlines()
                        print(f"Load vocabulary from {vocab_path} last chance...")
                except Exception as _:
                    print(f"No vocabulary was loaded at {vocab_path}")

        return sorted(_vocab)

    def process_vocabulary_from_doc(self,
                                    content_file_list: list,
                                    stem=False,
                                    min_vocab=2,
                                    is_spanish=False,
                                    greater=3) -> list:
        total = len(content_file_list)
        pool_iter = total // 8
        pool_iter_list = list()
        for n, pool_content_file_list in enumerate(
                ClassFile.divide_chunks(content_file_list, pool_iter), 1):
            pool_arg = (n, pool_content_file_list, stem, is_spanish, greater)
            pool_iter_list.append(pool_arg)

        # preprocess vocabulary
        pre_process_list = list()
        with Pool() as pool:
            # call the same function with different data in parallel
            for n, result in enumerate(
                    pool.starmap(self._process_vocabulary_from, pool_iter_list)):
                pre_process_list.extend(result)
                print(f"Finished process {n}")

        # filter word frequency
        dic_token_list = self.filter_token_min_occurrence(
            pre_process_list, min_occur=min_vocab)

        return sorted(list(set(dic_token_list)))

    def process_vocabulary_from_file(self,
                                     doc_file_list: list,
                                     stem=False,
                                     min_vocab=2,
                                     is_spanish=True,
                                     greater=3) -> list:
        print('Loading doc content...')
        full_text_list = list()
        for n, doc in enumerate(doc_file_list, 1):
            try:
                content = ClassFile.get_text(doc, encoding=self.ENCODING)
                if content:
                    full_text_list.append(content)
            except Exception as _:
                pass

        total = len(full_text_list)
        pool_iter = total // 100
        pool_iter_list = list()
        for n, pool_content_file_list in enumerate(
                ClassFile.divide_chunks(full_text_list, pool_iter), 1):
            pool_arg = (n, pool_content_file_list, stem, is_spanish, greater)
            pool_iter_list.append(pool_arg)

        # preprocess vocabulary
        pre_process_list = list()
        with Pool() as pool:
            # call the same function with different data in parallel
            for n, result in enumerate(
                    pool.starmap(self._process_vocabulary_from, pool_iter_list)):
                pre_process_list.extend(result)
                print(f"Finished process {n}")

        # filter word frequency
        dic_token_list = self.filter_token_min_occurrence(
            pre_process_list, min_occur=min_vocab)

        return sorted(list(set(dic_token_list)))

    def _process_vocabulary_from(self,
                                 process_id: int,
                                 content_list: list,
                                 stem=False,
                                 is_spanish=False,
                                 greater=3) -> list:
        pre_process_list = list()
        total = len(content_list)

        print(f"Total of documents: {total}")
        for n, content in enumerate(content_list, 1):
            print(f"Pre-processing [{process_id}/{n}/{total}]", end='')
            try:
                full_text = self.load_document(content)
                token_list = self.process_to_class(
                    full_text, stem=stem, is_spanish=is_spanish, greater=greater)
                if token_list:
                    print(f' -> ADDED {len(token_list)}', end='')
                    pre_process_list.extend(token_list)
                else:
                    print(' - > EMPTY', end='')
            except Exception as _:
                print(f' - > ERROR [{_}]', end='')
            print()

        return pre_process_list

    @staticmethod
    def load_document(content) -> str:
        try:
            full_text = content.decode(TextPreprocess.ENCODING, errors='replace')
        except (UnicodeDecodeError, AttributeError):
            full_text = content.encode(TextPreprocess.ENCODING, errors="replace").decode(
                TextPreprocess.ENCODING, errors='replace')
        try:
            data_encoded = full_text.encode(TextPreprocess.ENCODING, errors="replace")
            b64_encoded = base64.b64encode(data_encoded)
            b64_decoded = base64.b64decode(b64_encoded)
            full_text = b64_decoded.decode(TextPreprocess.ENCODING, errors='replace')
            try:
                if base64.b64encode(base64.b64decode(data_encoded)) == data_encoded:
                    b64_decoded = base64.b64decode(b64_decoded)
                    full_text = b64_decoded.decode(TextPreprocess.ENCODING, errors='replace')
            except Exception as _:
                pass
        except Exception as _:
            pass

        return full_text
