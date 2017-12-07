package de.hhu.mabre.languagetool;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class PythonDict {
    private List<NGram> nGrams = new LinkedList<>();
    private List<Integer> groundTruths = new LinkedList<>();

    void add(NGram nGram, int groundTruth) {
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
        int categories = Collections.max(groundTruths) + 1;
        return groundTruths.stream().map(i -> oneHotEncode(i, categories)).collect(Collectors.joining(","));
    }

    private static String oneHotEncode(int n, int categories) {
        String list = IntStream.range(0, categories).mapToObj(i -> n == i ? "1" : "0").collect(Collectors.joining(","));
        return "[" + list + "]";
    }
}
