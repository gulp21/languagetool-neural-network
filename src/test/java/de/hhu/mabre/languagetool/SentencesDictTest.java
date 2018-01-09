package de.hhu.mabre.languagetool;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class SentencesDictTest {

    @Test
    public void toJsonTest() {
        SentencesDict dict = new SentencesDict(2);
        dict.add(Collections.singletonList("Sie"), Arrays.asList("schlau", "."), 0);
        dict.add(Arrays.asList("Die", "beiden"), Arrays.asList("schlau", "."), 1);
        assertEquals("{\"tokensBefore\":[[\"Sie\"],[\"Die\",\"beiden\"]],\"tokensAfter\":[[\"schlau\",\".\"],[\"schlau\",\".\"]],\"groundTruths\":[[1,0],[0,1]],\"nCategories\":2}",
                dict.toJson());
    }

}