from typing import Union, Iterable

from py4j.java_gateway import JavaGateway, JavaObject
from py4j.java_collections import JavaList, ListConverter
from py4j.protocol import Py4JNetworkError


class LanguageTool:

    def __init__(self, languageCode):
        """
        Parameters
        ----------
        languageCode: code like "de-DE" or "en"
        """
        try:
            self.gateway = JavaGateway()
            self.languageCode = languageCode
            self.language = self.gateway.jvm.org.languagetool.Languages.getLanguageForShortCode(languageCode)
        except Py4JNetworkError as e:
            raise RuntimeError("Could not connect to JVM. Is ./gradlew pythonGateway running?") from e

    def tokenized_sentences(self, sentences: str) -> JavaList:
        """
        Split a string into sentences, and each sentences into tokens using LanguageTool.
        """
        return self.gateway.jvm.de.hhu.mabre.languagetool.FileTokenizer.tokenizedSentences(self.languageCode, sentences)

    def tokenize(self, sentences: str) -> JavaList:
        """
        Tokenize one or several sentences using LanguageTool.
        """
        return self.gateway.jvm.de.hhu.mabre.languagetool.FileTokenizer.tokenize(self.languageCode, sentences)

    @staticmethod
    def _get_tags_of_tagged_tokens(taggedToken: JavaObject):
        return list(map(lambda reading: reading.getPOSTag(), taggedToken.getReadings()))

    def tag(self, tokenizedSentences: Union[Iterable[str], JavaList]) -> [(str, [str])]:
        """
        Tag a tokenized text using the tagger of a language. All valid tags for a each token are returned.
        """
        if type(tokenizedSentences) is not JavaList:
            tokens = ListConverter().convert(tokenizedSentences, self.gateway._gateway_client)
        else:
            tokens = tokenizedSentences
        return list(zip(tokens, map(LanguageTool._get_tags_of_tagged_tokens, self.language.getTagger().tag(tokens))))
