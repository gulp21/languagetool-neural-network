package de.hhu.mabre.languagetool.transformationrules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;

public class Das_Dass implements TransformationRule {
    public boolean applicable(List<String> tokenizedSentence) {
        return randomIndexOfPattern(tokenizedSentence, Arrays.asList("das")).isPresent();
    }

    public List<String> apply(List<String> tokenizedSentence) {
        OptionalInt idx = randomIndexOfPattern(tokenizedSentence, Arrays.asList("das"));
        if (idx.isPresent()) {
            int i = idx.getAsInt();
            List<String> erroneousSentence = new ArrayList<>(tokenizedSentence);
            erroneousSentence.set(i, "dass");
            return erroneousSentence;
        }
        throw new IllegalArgumentException(this.getClass().toString() + " cannot be applied to " + String.join(" ", tokenizedSentence));
    }
}
