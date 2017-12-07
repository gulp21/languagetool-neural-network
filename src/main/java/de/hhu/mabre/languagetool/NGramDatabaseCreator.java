package de.hhu.mabre.languagetool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

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
        if(args.length < 5) {
            System.out.println("parameters: language-code training-filename validation-filename token1 token2");
            System.exit(-1);
        }

        String languageCode = args[0];
        String trainingFilename = args[1];
        String validationFilename = args[2];
        List<String> subjects = Arrays.asList(args).subList(3, args.length);

        writeDatabase(databaseFromSentences(languageCode, readText(trainingFilename), subjects, UNDERSAMPLE),trainingFilename+".py");
        writeDatabase(databaseFromSentences(languageCode, readText(validationFilename), subjects, NONE), validationFilename+".py");
    }

    private static void writeDatabase(PythonDict pythonDict, String filename) {
        try {
            Files.write(Paths.get(filename), Collections.singletonList(pythonDict.toString()));
            System.out.println(filename + " created");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static PythonDict databaseFromSentences(String languageCode, String sentences, List<String> subjects, SamplingMode samplingMode) {
        List<String> tokens = tokenize(languageCode, sentences);
        tokens.add(0, ".");
        tokens.add(1, ".");
        tokens.add(".");
        tokens.add(".");
        List<List<String>> tokenizedSubjects = new ArrayList<>();
        for (String subject: subjects) {
            tokenizedSubjects.add(tokenize(languageCode, subject));
        }
        return createDatabase(tokens, tokenizedSubjects, samplingMode);
    }

    static PythonDict databaseFromSentences(String languageCode, String sentences, String subject1, String subject2, SamplingMode samplingMode) {
        return databaseFromSentences(languageCode, sentences, Arrays.asList(subject1, subject2), samplingMode);
    }

    static PythonDict createDatabase(List<String> tokens, String token1, String token2, SamplingMode samplingMode) {
        return createDatabase(tokens, Arrays.asList(Collections.singletonList(token1), Collections.singletonList(token2)), samplingMode);
    }

    static PythonDict createDatabase(List<String> tokens, List<List<String>> subjects, SamplingMode samplingMode) {
        ArrayList<ArrayList<NGram>> nGrams = new ArrayList<>();

        for (List<String> subject: subjects) {
            nGrams.add(getRelevantNGrams(tokens, subject));
        }

        PythonDict db = new PythonDict();

        if(samplingMode == NONE) {
            for (int i = 0; i < nGrams.size(); i++) {
                db.addAll(nGrams.get(i), i);
            }
            return db;
        }

        int numberOfSamples = getNumberOfSamples(nGrams.stream().map(ArrayList::size).collect(Collectors.toList()), samplingMode);
        System.out.println("sampling to " + numberOfSamples);

        for (int i = 0; i < numberOfSamples; i++) {
            for (int n = 0; n < nGrams.size(); n++) {
                db.add(nGrams.get(n).get(i % nGrams.get(n).size()), n);
            }
        }
        return db;
    }

    private static int getNumberOfSamples(List<Integer> sampleCounts, SamplingMode samplingMode) {
        switch (samplingMode) {
            case NONE:
                throw new UnsupportedOperationException("NONE not supported here.");
            case UNDERSAMPLE:
                return Collections.min(sampleCounts);
            case OVERSAMPLE:
                return Collections.max(sampleCounts);
            case MODERATE_OVERSAMPLE:
                return Math.min(2 * Collections.min(sampleCounts), Collections.max(sampleCounts));
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
