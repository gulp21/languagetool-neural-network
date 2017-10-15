package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

import static de.hhu.mabre.languagetool.FileTokenizer.tokenize;

public class FileTokenizerTest extends TestCase {

    public void testTokenize() {
        String result = tokenize("en", "You’re not  here.");
        assertEquals("You ’ re not here .", result);
    }

}
