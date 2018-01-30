package de.hhu.mabre.languagetool.transformationrules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;

public class Dass_Das implements TransformationRule {
    public boolean applicable(List<String> tokenizedSentence) {
        return randomIndexOfPattern(tokenizedSentence, Arrays.asList("dass")).isPresent();
    }

    public List<String> apply(List<String> tokenizedSentence) {
        OptionalInt idx = randomIndexOfPattern(tokenizedSentence, Arrays.asList("dass"));
        if (idx.isPresent()) {
            int i = idx.getAsInt();
            List<String> erroneousSentence = new ArrayList<>(tokenizedSentence);
            erroneousSentence.set(i, "das");
            return erroneousSentence;
        }
        throw new IllegalArgumentException(this.getClass().toString() + " cannot be applied to " + String.join(" ", tokenizedSentence));
    }
}
