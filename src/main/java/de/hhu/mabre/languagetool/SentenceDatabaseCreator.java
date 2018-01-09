package de.hhu.mabre.languagetool;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static de.hhu.mabre.languagetool.FileTokenizer.tokenizedSentences;
import static de.hhu.mabre.languagetool.FileTokenizer.tokenize;
import static de.hhu.mabre.languagetool.SamplingMode.NONE;
import static de.hhu.mabre.languagetool.SamplingMode.UNDERSAMPLE;
import static de.hhu.mabre.languagetool.SubjectGrepper.grep;
import static de.hhu.mabre.languagetool.SubsetType.TRAINING;
import static de.hhu.mabre.languagetool.SubsetType.VALIDATION;

/**
 * Create a database with sentence starts for the neural network.
 */
public class SentenceDatabaseCreator {

    public static void main(String[] args) {
        if(args.length < 4) {
            System.out.println("parameters: language-code corpus subject1 subject2 …");
            System.exit(-1);
        }

        String languageCode = args[0];
        String corpusFilename = args[1];
        List<String> subjects = Arrays.asList(args).subList(2, args.length);

        List<String> relevantLines = new ArrayList<>(0);
        try {
            relevantLines = grep(new FileReader(corpusFilename), subjects.toArray(new String[subjects.size()]));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        List<List<String>> relevantSentences = tokenizedSentences(languageCode, String.join("\n", relevantLines));

        EnumMap<SubsetType, List<List<String>>> sets = randomlySplit(relevantSentences, 20);

        String basename = System.getProperty("java.io.tmpdir") + File.separator + String.join("_", subjects);

        writeDatabase(databaseFromSentences(languageCode, sets.get(TRAINING), subjects, UNDERSAMPLE), basename + "_training.json");
        writeDatabase(databaseFromSentences(languageCode, sets.get(VALIDATION), subjects, NONE), basename +"_validate.json");
    }

    static <T> EnumMap<SubsetType, List<T>> randomlySplit(List<T> items, int validatePercentage) {
        Collections.shuffle(items);
        int totalLines = items.size();
        int firstTrainingIndex = validatePercentage * totalLines / 100;
        List<T> trainingLines = items.subList(firstTrainingIndex, totalLines);
        List<T> validationLines = items.subList(0, firstTrainingIndex);
        EnumMap<SubsetType, List<T>> sets = new EnumMap<>(SubsetType.class);
        sets.put(TRAINING, trainingLines);
        sets.put(VALIDATION, validationLines);
        return sets;
    }

    private static void writeDatabase(SentencesDict dict, String filename) {
        try {
            Files.write(Paths.get(filename), Collections.singletonList(dict.toJson()));
            System.out.println(filename + " created");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static SentencesDict databaseFromSentences(String languageCode, List<List<String>> tokenizedSentences, List<String> subjects, SamplingMode samplingMode) {
        List<List<String>> tokenizedSubjects = new ArrayList<>();
        for (String subject: subjects) {
            tokenizedSubjects.add(tokenize(languageCode, subject));
        }
        return createDatabase(tokenizedSentences, tokenizedSubjects, samplingMode);
    }

    static SentencesDict createDatabase(List<List<String>> tokenizedSentences, List<List<String>> subjects, SamplingMode samplingMode) {
        List<List<List<String>>> sentenceStarts = new ArrayList<>();
        List<List<List<String>>> sentenceEndings = new ArrayList<>();

        for (List<String> subject: subjects) {
            sentenceStarts.add(getRelevantSentenceBeginnings(tokenizedSentences, subject));
            sentenceEndings.add(getRelevantSentenceEndings(tokenizedSentences, subject));
        }

        SentencesDict db = new SentencesDict(subjects.size());

        List<Integer> numberOfSamples = getNumberOfSamples(sentenceStarts.stream().map(List::size).collect(Collectors.toList()), samplingMode);
        System.out.println("sampling to " + Arrays.toString(numberOfSamples.toArray()));

        for (int n = 0; n < sentenceStarts.size(); n++) {
            for (int i = 0; i < numberOfSamples.get(n); i++) {
                db.add(sentenceStarts.get(n).get(i % sentenceStarts.get(n).size()),
                       sentenceEndings.get(n).get(i % sentenceEndings.get(n).size()),
                       n);
            }
        }
        return db;
    }

    private static List<Integer> getNumberOfSamples(List<Integer> sampleCounts, SamplingMode samplingMode) {
        if (samplingMode != NONE) {
            int numberOfSamples = 0;
            switch (samplingMode) {
                case UNDERSAMPLE:
                    numberOfSamples = Collections.min(sampleCounts);
                    break;
                case OVERSAMPLE:
                    numberOfSamples = Collections.max(sampleCounts);
                    break;
                case MODERATE_OVERSAMPLE:
                    numberOfSamples = Math.min(2 * Collections.min(sampleCounts), Collections.max(sampleCounts));
                    break;
            }
            return Collections.nCopies(sampleCounts.size(), numberOfSamples);
        }
        return sampleCounts;
    }

    static List<List<String>> getRelevantSentenceBeginnings(List<List<String>> tokenizedSentences, List<String> subjectTokens) {
        List<List<String>> nGrams = new ArrayList<>();
        final int subjectLength = subjectTokens.size();
        for (List<String> tokens : tokenizedSentences) {
            for (int i = 0; i <= tokens.size() - subjectLength; i++) {
                if (tokens.subList(i, i + subjectLength).equals(subjectTokens)) {
                    nGrams.add(tokens.subList(0, i));
                }
            }
        }
        return nGrams;
    }

    static List<List<String>> getRelevantSentenceEndings(List<List<String>> tokenizedSentences, List<String> subjectTokens) { // parameterize
        List<List<String>> nGrams = new ArrayList<>();
        final int subjectLength = subjectTokens.size();
        for (List<String> tokens : tokenizedSentences) {
            for (int i = 0; i <= tokens.size() - subjectLength; i++) {
                if (tokens.subList(i, i + subjectLength).equals(subjectTokens)) {
                    nGrams.add(tokens.subList(i + 1, tokens.size()));
                }
            }
        }
        return nGrams;
    }

}
