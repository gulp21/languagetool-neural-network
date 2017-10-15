package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

import java.util.Arrays;
import java.util.List;

import static de.hhu.mabre.languagetool.FileTokenizer.tokenize;

public class FileTokenizerTest extends TestCase {

    public void testTokenize() {
        List<String> result = tokenize("en", "You’re not  here.");
        assertEquals(Arrays.asList("You", "’", "re", "not", "here", "."), result);
    }

}
