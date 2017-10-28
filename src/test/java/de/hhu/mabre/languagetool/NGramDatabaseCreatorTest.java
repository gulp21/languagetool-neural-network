package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

import java.util.Arrays;
import java.util.List;

import static de.hhu.mabre.languagetool.NGramDatabaseCreator.createDatabase;
import static de.hhu.mabre.languagetool.NGramDatabaseCreator.databaseFromSentences;
import static de.hhu.mabre.languagetool.NGramDatabaseCreator.getRelevantNGrams;

public class NGramDatabaseCreatorTest extends TestCase {

    public void testGetRelevantNGrams() {
        List<NGram> nGrams = getRelevantNGrams(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i"), "c");
        List<NGram> expectedNGrams = Arrays.asList(
                new NGram("c", "b", "c", "d", "e"),
                new NGram("g", "h", "c", "c", "i"));
        assertEquals(expectedNGrams, nGrams);
    }

    public void testCreateDatabase() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i"), "c", "f", SamplingMode.UNDERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseModerateOversample() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.MODERATE_OVERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['d','e','f','g','h'],['g','h','c','c','i'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[0,1],[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseOversample() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.OVERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['d','e','f','g','h'],['g','h','c','c','i'],['d','e','f','g','h'],['h','c','c','i','j'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseNoSampling() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.NONE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['g','h','c','c','i'],['h','c','c','i','j'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[1,0],[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testDatabaseFromSentences() {
        PythonDict db = databaseFromSentences("en", "I like that, too. I would like to go to the museum, too.", "to", "too");
        String expectedDb = "{'ngrams':[['would','like','to','go','to'],['that',',','too','.','I']],\n'groundtruths':[[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testDatabaseFromSentencesSingleQuoteEscaping() {
        PythonDict db = databaseFromSentences("en", "Whare is 'The Station'? I would like to go to the museum.", "Station", "to");
        String expectedDb = "{'ngrams':[['\\'','The','Station','\\'','?'],['would','like','to','go','to']],\n'groundtruths':[[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

}
