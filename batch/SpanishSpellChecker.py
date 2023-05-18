import sys

from spellchecker import SpellChecker

from batch.SpacyService import SpacyService


class SpanishSpellChecker:
    def __init__(self):
        self.spell = SpellChecker(language='es', distance=2)

    def is_spanish(self, doc) -> (bool, float):
        if not doc:
            return False, 0

        work_list = self.tokenizer(doc)

        # find those words that may be misspelled
        misspelled = self.spell.unknown(work_list)
        spelled = self.spell.known(work_list)

        ratio = len(spelled) / (len(misspelled) + sys.float_info.min)
        return len(spelled) >= len(misspelled), ratio

    def tokenizer(self, sentence):
        token_list = list()
        clean_sentence = self.speller_clean(sentence)
        nlp = SpacyService.getInstance()
        doc = nlp.analyze(clean_sentence)
        for token in doc:
            if (
                    token.pos_ == r'PROPN' or
                    token.pos_ == r'NOUN' or
                    token.pos_ == r'ADJ' or
                    token.pos_ == r'ADV' or
                    token.pos_ == r'AUX' or
                    token.pos_ == r'INTJ' or
                    token.pos_ == r'VERB' or
                    token.pos_ == r'ADP' or
                    token.pos_ == r'DET'
            ):
                token_list.append(token.text)

        return token_list

    @staticmethod
    def speller_clean(data):
        if data is None:
            return ''

        if type(data) == int or type(data) == float:
            return data

        data = data.strip()
        data = data.replace('\r', ' ')
        data = data.replace('\n', ' ')
        data = data.replace('\t', ' ')
        data = data.replace('[', ' ')
        data = data.replace(']', ' ')
        data = data.replace('/', ' ')
        data = data.replace('.', ' ')
        data = data.replace(':', ' ')
        data = data.replace(';', ' ')
        data = data.replace('\'', ' ')

        data = data.lower()
        return data
