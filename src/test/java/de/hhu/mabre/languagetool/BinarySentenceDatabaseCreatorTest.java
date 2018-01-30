package de.hhu.mabre.languagetool;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static de.hhu.mabre.languagetool.BinarySentenceDatabaseCreator.introduceKommaOhneDass_OhneDas;
import static org.junit.Assert.assertEquals;

public class BinarySentenceDatabaseCreatorTest {

    @Test
    public void introduceKommaOhneDass_OhneDasTest() throws Exception {
        List<String> correct = Arrays.asList("Sie", "können", "sich", "ändern", ",", "ohne", "dass", "man", "es", "merkt", ".");
        List<String> expected = Arrays.asList("Sie", "können", "sich", "ändern", "ohne", "das", "man", "es", "merkt", ".");
        List<String> actual = introduceKommaOhneDass_OhneDas(correct);
        assertEquals(expected, actual);
    }

}