package de.hhu.mabre.languagetool;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static de.hhu.mabre.languagetool.SentenceDatabaseCreator.createDatabase;
import static de.hhu.mabre.languagetool.SentenceDatabaseCreator.getRelevantSentenceBeginnings;
import static de.hhu.mabre.languagetool.SentenceDatabaseCreator.getRelevantSentenceEndings;
import static org.junit.Assert.assertEquals;

public class SentenceDatabaseCreatorTest {

    @Test
    public void testGetRelevantSentenceBeginningsWithSingleTokenSubject() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        List<List<String>> result = getRelevantSentenceBeginnings(sentences, Collections.singletonList("foo"));
        List<List<String>> expected = Arrays.asList(
                Arrays.asList("a", "b", "c"),
                Arrays.asList("a", "b"));
        assertEquals(expected, result);
    }

    @Test
    public void testGetRelevantSentenceBeginningsWithMultiTokenSubject() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo", "bar"),
                Arrays.asList("a", "b", "b", "foo", "bar", "d"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        List<List<String>> result = getRelevantSentenceBeginnings(sentences, Arrays.asList("foo", "bar"));
        List<List<String>> expected = Arrays.asList(
                Arrays.asList("a", "b", "c"),
                Arrays.asList("a", "b", "b"));
        assertEquals(expected, result);
    }

    @Test
    public void testGetRelevantSentenceEndingsWithSingleTokenSubject() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        List<List<String>> result = getRelevantSentenceEndings(sentences, Collections.singletonList("foo"));
        List<List<String>> expected = Arrays.asList(
                Collections.emptyList(),
                Collections.singletonList("c"));
        assertEquals(expected, result);
    }

    @Test
    public void createDatabaseTest() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo", "bar"),
                Arrays.asList("a", "b", "b", "foo", "bar", "d"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        SentencesDict result = createDatabase(sentences, Collections.singletonList(Collections.singletonList("foo")), SamplingMode.NONE);
        assertEquals("{\"tokensBefore\":[[\"a\",\"b\",\"c\"],[\"a\",\"b\",\"b\"],[\"a\",\"b\"]],\"tokensAfter\":[[\"bar\"],[\"bar\",\"d\"],[\"c\"]],\"groundTruths\":[[1],[1],[1]],\"nCategories\":1}", result.toJson());
    }

}