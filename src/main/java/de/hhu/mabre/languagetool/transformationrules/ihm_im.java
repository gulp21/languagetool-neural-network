package de.hhu.mabre.languagetool.transformationrules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;

public class ihm_im implements TransformationRule {
    public boolean applicable(List<String> tokenizedSentence) {
        return randomIndexOfPattern(tokenizedSentence, Arrays.asList("ihm")).isPresent();
    }

    public List<String> apply(List<String> tokenizedSentence) {
        OptionalInt idx = randomIndexOfPattern(tokenizedSentence, Arrays.asList("ihm"));
        if (idx.isPresent()) {
            int i = idx.getAsInt();
            List<String> erroneousSentence = new ArrayList<>(tokenizedSentence);
            erroneousSentence.set(i, "im");
            return erroneousSentence;
        }
        throw new IllegalArgumentException(this.getClass().toString() + " cannot be applied to " + String.join(" ", tokenizedSentence));
    }
}
