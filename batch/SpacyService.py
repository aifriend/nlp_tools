import es_core_news_sm

from common.Configuration import Configuration
from common.commonsLib import loggerElk

logger = loggerElk(__name__)


class SpacyService:
    __instance = None
    __nlp = None
    conf = Configuration()

    @staticmethod
    def getInstance():
        """ Static access method. """
        if not SpacyService.__instance:
            SpacyService()
        return SpacyService.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if SpacyService.__instance:
            raise FileNotFoundError("This class is a singleton!")
        else:
            SpacyService.__nlp = es_core_news_sm.load()
            SpacyService.__instance = self

    def analyze(self, doc):
        return self.__nlp(doc)
