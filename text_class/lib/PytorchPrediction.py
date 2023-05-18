import os

from text_class.lib.PytorchChecker import PytorchChecker


class PytorchPrediction:
    __instance = None
    __ner = None

    @staticmethod
    def getInstance(conf=None):
        """ Static access method. """
        if not PytorchPrediction.__instance:
            PytorchPrediction(conf)
        return PytorchPrediction.__instance

    @staticmethod
    def getChecker():
        return PytorchPrediction.__ner

    def __init__(self, conf=None):
        """ Virtually private constructor. """
        if PytorchPrediction.__instance:
            raise IOError("PytorchPrediction::Class is a singleton!")
        else:
            if conf:
                model_path = os.path.join(
                    conf.root_path, conf.bert_model_path, conf.bert_ner_name)
                tokenizer_path = os.path.join(
                    conf.root_path, conf.bert_model_path, conf.bert_tokenizer_path)
                PytorchPrediction.__ner = PytorchChecker(model_path, tokenizer_path)
                PytorchPrediction.__instance = self
            else:
                raise ValueError("PytorchPrediction::Model path not found")

    def analyze(self, doc: list):
        return self.__ner.predict(doc)
