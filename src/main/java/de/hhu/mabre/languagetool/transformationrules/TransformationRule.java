package de.hhu.mabre.languagetool.transformationrules;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

public interface TransformationRule {

    boolean applicable(List<String> tokenizedSentence);

    List<String> apply(List<String> tokenizedSentence);

    default OptionalInt randomIndexOfPattern(List<String> tokenizedSentence, List<String> pattern) {
        ArrayList<Integer> matchingIndices = new ArrayList<>();
        int patternSize = pattern.size();
        for (int i = 0; i < tokenizedSentence.size() - patternSize; i++) {
            if (tokenizedSentence.subList(i, i + patternSize).equals(pattern)) {
                matchingIndices.add(i);
            }
        }
        if (matchingIndices.isEmpty()) {
            return OptionalInt.empty();
        } else {
            return OptionalInt.of(matchingIndices.get((int) (Math.random() * matchingIndices.size())));
        }
    }
}
