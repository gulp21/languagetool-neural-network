package de.hhu.mabre.languagetool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static de.hhu.mabre.languagetool.FileTokenizer.readText;
import static de.hhu.mabre.languagetool.FileTokenizer.tokenize;

/**
 * Create a 5-gram database as input for the neural network.
 */
public class NGramDatabaseCreator {

    private static final int N = 5;

    public static void main(String[] args) {
        if(args.length != 5) {
            System.out.println("parameters: language-code training-filename validation-filename token1 token2");
            System.exit(-1);
        }

        String languageCode = args[0];
        String trainingFilename = args[1];
        String validationFilename = args[2];
        String token1 = args[3];
        String token2 = args[4];

        writeDatabase(databaseFromSentences(languageCode, readText(trainingFilename), token1, token2),trainingFilename+".py");
        writeDatabase(databaseFromSentences(languageCode, readText(validationFilename), token1, token2), validationFilename+".py");
    }

    private static void writeDatabase(PythonDict pythonDict, String filename) {
        try {
            Files.write(Paths.get(filename), Collections.singletonList(pythonDict.toString()));
            System.out.println(filename + "created");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static PythonDict databaseFromSentences(String languageCode, String sentences, String token1, String token2) {
        List<String> tokens = tokenize(languageCode, sentences);
        return createDatabase(tokens, token1, token2);
    }

    static PythonDict createDatabase(List<String> tokens, String token1, String token2) {
        ArrayList<NGram> token1NGrams = getRelevantNGrams(tokens, token1);
        ArrayList<NGram> token2NGrams = getRelevantNGrams(tokens, token2);

        int numberOfSamples = Math.min(token1NGrams.size(), token2NGrams.size());
        System.out.println("undersampling to " + numberOfSamples);

        PythonDict db = new PythonDict();
        token1NGrams.stream().limit(numberOfSamples).forEach(nGram -> db.add(nGram, 0));
        token2NGrams.stream().limit(numberOfSamples).forEach(nGram -> db.add(nGram, 1));
        return db;
    }

    static ArrayList<NGram> getRelevantNGrams(List<String> tokens, String token) {
        ArrayList<NGram> nGrams;
        nGrams = new ArrayList<>();
        int end = tokens.size() - N/2;
        for(int i = N/2; i < end; i++) {
            if (tokens.get(i).equals(token)) {
                nGrams.add(new NGram(tokens.subList(i-N/2, i+N/2+1)));
            }
        }
        return nGrams;
    }

}
