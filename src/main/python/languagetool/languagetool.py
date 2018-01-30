from typing import Union, Iterable

from py4j.java_gateway import JavaGateway, JavaObject
from py4j.java_collections import JavaList, ListConverter
from py4j.protocol import Py4JNetworkError

try:
    gateway = JavaGateway()
    english = gateway.jvm.org.languagetool.language.English()
    german = gateway.jvm.org.languagetool.language.German()
except Py4JNetworkError as e:
    raise RuntimeError("Could not connect to JVM. Is ./gradlew pythonGateway running?") from e


def tokenized_sentences(languageCode: str, sentences: str) -> JavaList:
    """
    Split a string into sentences, and each sentences into tokens using LanguageTool.

    Parameters
    ----------
    languageCode: code like "de-DE"
    sentences
    """
    return gateway.jvm.de.hhu.mabre.languagetool.FileTokenizer.tokenizedSentences(languageCode, sentences)


def tokenize(languageCode: str, sentences: str) -> JavaList:
    """
    Tokenize one or several sentences using LanguageTool.

    Parameters
    ----------
    languageCode: code like "de-DE"
    """
    return gateway.jvm.de.hhu.mabre.languagetool.FileTokenizer.tokenize(languageCode, sentences)


def _get_tags_of_tagged_tokens(taggedToken: JavaObject):
    return list(map(lambda reading: reading.getPOSTag(), taggedToken.getReadings()))


def tag(language: JavaObject, tokenizedSentences: Union[Iterable[str], JavaList]) -> [(str, [str])]:
    """
    Tag a tokenized text using the tagger of a language. All valid tags for a each token are returned.

    Parameters
    ----------
    language: language instance from org.languagetool.language package
    """
    if type(tokenizedSentences) is not JavaList:
        tokens = ListConverter().convert(tokenizedSentences, gateway._gateway_client)
    else:
        tokens = tokenizedSentences
    return list(zip(tokens, map(_get_tags_of_tagged_tokens, language.getTagger().tag(tokens))))
