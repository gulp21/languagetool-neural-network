package de.hhu.mabre.languagetool;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static de.hhu.mabre.languagetool.SentenceDatabaseCreator.getRelevantSentenceBeginnings;
import static org.junit.Assert.assertEquals;

public class SentenceDatabaseCreatorTest {

    @Test
    public void testGetRelevantSentenceBeginningsWithSingleTokenSubject() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        ArrayList<NGram> result = getRelevantSentenceBeginnings(sentences, Collections.singletonList("foo"));
        List<NGram> expected = Arrays.asList(
                new NGram("a", "b", "c"),
                new NGram("a", "b"));
        assertEquals(expected, result);
    }

    @Test
    public void testGetRelevantSentenceBeginningsWithMultiTokenSubject() {
        List<List<String>> sentences = Arrays.asList(
                Arrays.asList("a", "b", "c", "foo", "bar"),
                Arrays.asList("a", "b", "b", "foo", "bar", "d"),
                Arrays.asList("a", "b", "c", "bar", "d"),
                Arrays.asList("a", "b", "foo", "c"));
        ArrayList<NGram> result = getRelevantSentenceBeginnings(sentences, Arrays.asList("foo", "bar"));
        List<NGram> expected = Arrays.asList(
                new NGram("a", "b", "c"),
                new NGram("a", "b", "b"));
        assertEquals(expected, result);
    }

}