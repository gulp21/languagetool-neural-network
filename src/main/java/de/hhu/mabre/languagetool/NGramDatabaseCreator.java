package de.hhu.mabre.languagetool;

import com.google.common.collect.Lists;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static de.hhu.mabre.languagetool.FileTokenizer.readText;
import static de.hhu.mabre.languagetool.FileTokenizer.tokenize;
import static de.hhu.mabre.languagetool.SamplingMode.NONE;
import static de.hhu.mabre.languagetool.SamplingMode.UNDERSAMPLE;

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

        writeDatabase(databaseFromSentences(languageCode, readText(trainingFilename), token1, token2, UNDERSAMPLE),trainingFilename+".py");
        writeDatabase(databaseFromSentences(languageCode, readText(validationFilename), token1, token2, NONE), validationFilename+".py");
    }

    private static void writeDatabase(PythonDict pythonDict, String filename) {
        try {
            Files.write(Paths.get(filename), Collections.singletonList(pythonDict.toString()));
            System.out.println(filename + "created");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static PythonDict databaseFromSentences(String languageCode, String sentences, String token1, String token2, SamplingMode samplingMode) {
        List<String> tokens = tokenize(languageCode, sentences);
        tokens.add(0, ".");
        tokens.add(1, ".");
        tokens.add(".");
        tokens.add(".");
        List<String> tokens1 = tokenize(languageCode, token1);
        List<String> tokens2 = tokenize(languageCode, token2);
        return createDatabase(tokens, tokens1, tokens2, samplingMode);
    }

    static PythonDict createDatabase(List<String> tokens, String token1, String token2, SamplingMode samplingMode) {
        return createDatabase(tokens, Collections.singletonList(token1), Collections.singletonList(token2), samplingMode);
    }

    static PythonDict createDatabase(List<String> tokens, List<String> tokens1, List<String> tokens2, SamplingMode samplingMode) {
        ArrayList<NGram> token1NGrams = getRelevantNGrams(tokens, tokens1);
        ArrayList<NGram> token2NGrams = getRelevantNGrams(tokens, tokens2);

        int token1size = token1NGrams.size();
        int token2Count = token2NGrams.size();

        PythonDict db = new PythonDict();

        if(samplingMode == NONE) {
            db.addAll(token1NGrams, 0);
            db.addAll(token2NGrams, 1);
            return db;
        }

        int numberOfSamples = getNumberOfSamples(token1size, token2Count, samplingMode);
        System.out.println("sampling to " + numberOfSamples);

        for (int i = 0; i < numberOfSamples; i++) {
            db.add(token1NGrams.get(i % token1size), 0);
            db.add(token2NGrams.get(i % token2Count), 1);
        }
        return db;
    }

    private static int getNumberOfSamples(int token1size, int token2Count, SamplingMode samplingMode) {
        switch (samplingMode) {
            case NONE:
                throw new UnsupportedOperationException("NONE not supported here.");
            case UNDERSAMPLE:
                return Math.min(token1size, token2Count);
            case OVERSAMPLE:
                return Math.max(token1size, token2Count);
            case MODERATE_OVERSAMPLE:
                return Math.min(Math.min(2*token1size, 2*token2Count), Math.max(token1size, token2Count));
        }
        return -1;
    }

    static ArrayList<NGram> getRelevantNGrams(List<String> tokens, List<String> subjectTokens) {
        ArrayList<NGram> nGrams;
        nGrams = new ArrayList<>();

        final int end = tokens.size() - N/2;
        final int subjectLength = subjectTokens.size();
        for(int i = N/2 - 1; i <= end - subjectLength; i++) {
            if (tokens.subList(i, i+subjectLength).equals(subjectTokens)) {
                List<String> ngram = new LinkedList<>();
                ngram.addAll(tokens.subList(i-N/2, i));
                ngram.add(subjectTokens.stream().collect(Collectors.joining(" ")));
                ngram.addAll(tokens.subList(i+subjectLength, i+N/2+subjectLength));
                nGrams.add(new NGram(ngram));
            }
        }
        return nGrams;
    }

}
