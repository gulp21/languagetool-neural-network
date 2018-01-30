package de.hhu.mabre.languagetool.transformationrules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;

public class im_ihm implements TransformationRule {
    public boolean applicable(List<String> tokenizedSentence) {
        return randomIndexOfPattern(tokenizedSentence, Arrays.asList("im")).isPresent();
    }

    public List<String> apply(List<String> tokenizedSentence) {
        OptionalInt idx = randomIndexOfPattern(tokenizedSentence, Arrays.asList("im"));
        if (idx.isPresent()) {
            int i = idx.getAsInt();
            List<String> erroneousSentence = new ArrayList<>(tokenizedSentence);
            erroneousSentence.set(i, "ihm");
            return erroneousSentence;
        }
        throw new IllegalArgumentException(this.getClass().toString() + " cannot be applied to " + String.join(" ", tokenizedSentence));
    }
}
