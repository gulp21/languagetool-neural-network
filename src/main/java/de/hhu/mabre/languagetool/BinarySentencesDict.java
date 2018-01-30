package de.hhu.mabre.languagetool;

import com.google.gson.Gson;

import java.util.LinkedList;
import java.util.List;

class BinarySentencesDict {
    private final List<List<String>> tokens = new LinkedList<>();
    private final List<Integer> groundTruths = new LinkedList<>();

    void add(List<String> tokens, boolean correct) {
        this.tokens.add(tokens);
        groundTruths.add(correct ? 1 : -1);
    }

    public String toJson() {
        return new Gson().toJson(this);
    }
}
