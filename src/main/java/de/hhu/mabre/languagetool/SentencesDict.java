package de.hhu.mabre.languagetool;

import com.google.gson.Gson;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.IntStream;

class SentencesDict {
    private final List<List<String>> tokensBefore = new LinkedList<>();
    private final List<List<String>> tokensAfter = new LinkedList<>();
    private final List<long[]> groundTruths = new LinkedList<>();
    private final int nCategories;

    SentencesDict(int nCategories) {
        this.nCategories = nCategories;
    }

    void add(List<String> tokensBefore, List<String> tokensAfter, int groundTruth) {
        this.tokensBefore.add(tokensBefore);
        this.tokensAfter.add(tokensAfter);
        groundTruths.add(oneHotEncode(groundTruth));
    }

    public String toJson() {
        return new Gson().toJson(this);
    }

    private long[] oneHotEncode(int n) {
        return IntStream.range(0, nCategories).mapToLong(i -> n == i ? 1 : 0).toArray();
    }
}
