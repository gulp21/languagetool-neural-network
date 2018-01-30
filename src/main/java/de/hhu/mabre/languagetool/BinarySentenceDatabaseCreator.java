package de.hhu.mabre.languagetool;

import de.hhu.mabre.languagetool.transformationrules.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static de.hhu.mabre.languagetool.FileTokenizer.tokenizedSentences;
import static de.hhu.mabre.languagetool.SubsetType.TRAINING;
import static de.hhu.mabre.languagetool.SubsetType.VALIDATION;

public class BinarySentenceDatabaseCreator {

    public static void main(String[] args) {
        if(args.length != 2) {
            System.out.println("parameters: language-code corpus");
            System.exit(-1);
        }

        String languageCode = args[0];
        String corpusFilename = args[1];

        List<String> lines = new ArrayList<>(0);
        try {
            lines = Files.readAllLines(Paths.get(corpusFilename));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        List<List<String>> relevantSentences = tokenizedSentences(languageCode, String.join("\n", lines));

        EnumMap<SubsetType, List<List<String>>> sets = randomlySplit(relevantSentences, 20);

        String basename = System.getProperty("java.io.tmpdir") + File.separator + "A";

        writeDatabase(createDatabase(sets.get(TRAINING)), basename + "_training.json");
        writeDatabase(createDatabase(sets.get(VALIDATION)), basename +"_validate.json");
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

    private static void writeDatabase(BinarySentencesDict dict, String filename) {
        try {
            Files.write(Paths.get(filename), Collections.singletonList(dict.toJson()));
            System.out.println(filename + " created");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static BinarySentencesDict createDatabase(List<List<String>> tokenizedSentences) {
        BinarySentencesDict db = new BinarySentencesDict();

        List<TransformationRule> transformationRules = Arrays.asList(
                new KommaDass_Das(),
                new KommaOhneDass_OhneDas(),
                new KommaDas_Das(),
                new Das_Dass(),
                new Dass_Das(),
                new im_ihm(),
                new ihm_im()
        );

        for (List<String> sentence: tokenizedSentences) {
            Collections.shuffle(transformationRules);
            Optional<TransformationRule> transformationRule = transformationRules.stream().filter(tr -> tr.applicable(sentence)).findFirst();
            if (transformationRule.isPresent()) {
                db.add(sentence, true);
                db.add(transformationRule.get().apply(sentence), false);
            }
        }

        return db;
    }
}
