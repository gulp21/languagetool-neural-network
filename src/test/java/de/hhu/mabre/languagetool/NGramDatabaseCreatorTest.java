package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import static de.hhu.mabre.languagetool.NGramDatabaseCreator.*;
import static de.hhu.mabre.languagetool.SamplingMode.NONE;
import static de.hhu.mabre.languagetool.SamplingMode.UNDERSAMPLE;
import static de.hhu.mabre.languagetool.SubsetType.TRAINING;
import static de.hhu.mabre.languagetool.SubsetType.VALIDATION;

public class NGramDatabaseCreatorTest extends TestCase {

    public void testGetRelevantNGrams() {
        List<NGram> nGrams = getRelevantNGrams(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i"), Collections.singletonList("c"));
        List<NGram> expectedNGrams = Arrays.asList(
                new NGram("c", "b", "c", "d", "e"),
                new NGram("g", "h", "c", "c", "i"));
        assertEquals(expectedNGrams, nGrams);
    }

    public void testGetRelevantNGramsWithMultiTokens() {
        List<NGram> nGrams = getRelevantNGrams(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i"), Arrays.asList("d", "e"));
        List<NGram> expectedNGrams = Collections.singletonList(new NGram("b", "c", "d e", "f", "g"));
        assertEquals(expectedNGrams, nGrams);
    }

    public void testGetRelevantNGramsWithMultiTokensAtBorder() {
        List<NGram> nGrams = getRelevantNGrams(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "c", "d", "c", "i"), Arrays.asList("c", "d"));
        List<NGram> expectedNGrams = Arrays.asList(
                new NGram("c", "b", "c d", "e", "f"),
                new NGram("f", "g", "c d", "c", "i"));
        assertEquals(expectedNGrams, nGrams);
    }

    public void testCreateDatabase() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i"), "c", "f", UNDERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseModerateOversample() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.MODERATE_OVERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['g','h','c','c','i'],['d','e','f','g','h'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[1,0],[0,1],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseOversample() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.OVERSAMPLE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['g','h','c','c','i'],['h','c','c','i','j'],['d','e','f','g','h'],['d','e','f','g','h'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testCreateDatabaseNoSampling() {
        PythonDict db = createDatabase(Arrays.asList("c", "b", "c", "d", "e", "f", "g", "h", "c", "c", "i", "j"), "c", "f", SamplingMode.NONE);
        String expectedDb = "{'ngrams':[['c','b','c','d','e'],['g','h','c','c','i'],['h','c','c','i','j'],['d','e','f','g','h']],\n'groundtruths':[[1,0],[1,0],[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testDatabaseFromSentences() {
        PythonDict db = databaseFromSentences("en", "I like that, too. I would like to go to the museum, too.", "to", "too", UNDERSAMPLE);
        String expectedDb = "{'ngrams':[['would','like','to','go','to'],['to','go','to','the','museum'],['that',',','too','.','I'],['museum',',','too','.','.']],\n'groundtruths':[[1,0],[1,0],[0,1],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testDatabaseFromSentencesSingleQuoteEscaping() {
        PythonDict db = databaseFromSentences("en", "Whare is 'The Station'? I would like to go to the museum.", "Station", "to", UNDERSAMPLE);
        String expectedDb = "{'ngrams':[['\\'','The','Station','\\'','?'],['would','like','to','go','to']],\n'groundtruths':[[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testMultiTokenSubjects() {
        PythonDict db = databaseFromSentences("de", "Ich habe das nicht gesagt. Ich habe gesagt, dass ich zum Karaokeabend kommen werde. Ich mag das nicht.", "das", ", dass", NONE);
        String expectedDb = "{'ngrams':[['Ich','habe','das','nicht','gesagt'],['Ich','mag','das','nicht','.'],['habe','gesagt',', dass','ich','zum']],\n'groundtruths':[[1,0],[1,0],[0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testDatabaseFromSentencesAndThreeTokens() {
        PythonDict db = databaseFromSentences("en", "I like that, too. I would like to go to two museums, too.", Arrays.asList("to", "too", "two"), UNDERSAMPLE);
        String expectedDb = "{'ngrams':[['would','like','to','go','to'],['that',',','too','.','I'],['go','to','two','museums',',']],\n'groundtruths':[[1,0,0],[0,1,0],[0,0,1]]}";
        assertEquals(expectedDb, db.toString());
    }

    public void testRandomlySplit() {
        List<String> lines = Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
        HashMap<SubsetType, String> sets = randomlySplit(lines, 32);
        assertEquals(3, sets.get(VALIDATION).split("\n").length);
        assertEquals(7, sets.get(TRAINING).split("\n").length);
    }

}
