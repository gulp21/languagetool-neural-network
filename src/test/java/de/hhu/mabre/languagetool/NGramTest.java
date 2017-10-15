package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

public class NGramTest extends TestCase {

    public void testEquals() {
        NGram nGram1 = new NGram("g", "h", "c", "i", "j");
        NGram nGram2 = new NGram("g", "h", "c", "i", "j");
        assertEquals(nGram1, nGram2);
    }

}
