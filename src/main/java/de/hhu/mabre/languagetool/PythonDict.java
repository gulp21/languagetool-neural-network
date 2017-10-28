package de.hhu.mabre.languagetool;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

class PythonDict {
    private List<NGram> nGrams = new LinkedList<>();
    private List<Integer> groundTruths = new LinkedList<>();

    void add(NGram nGram, int groundTruth) {
        if(groundTruth != 0 && groundTruth != 1) {
            throw new IllegalArgumentException("groundTruth must be 0 or 1");
        }
        nGrams.add(nGram);
        groundTruths.add(groundTruth);
    }

    void addAll(List<NGram> nGrams, int groundTruth) {
        nGrams.forEach(nGram -> add(nGram, groundTruth));
    }

    @Override
    public String toString() {
        String dict;
        dict = "{'ngrams':[";
        dict += nGrams.stream().map(NGram::toString).collect(Collectors.joining(","));
        dict += "],\n";
        dict += "'groundtruths':[" + oneHotEncode(groundTruths) + "]}";
        return dict;
    }

    private static String oneHotEncode(List<Integer> groundTruths) {
        return groundTruths.stream().map(i -> i == 0 ? "[1,0]" : "[0,1]").collect(Collectors.joining(","));
    }
}
