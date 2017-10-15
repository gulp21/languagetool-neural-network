package de.hhu.mabre.languagetool;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

class NGram {
    private List<String> tokens;

    NGram(List<String> tokens) {
        this.tokens = tokens;
    }

    NGram(String ...tokens) {
        this.tokens = Arrays.asList(tokens);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NGram nGram = (NGram) o;

        return tokens.equals(nGram.tokens);
    }

    @Override
    public int hashCode() {
        return tokens.hashCode();
    }

    @Override
    public String toString() {
        return "["
                + tokens.stream().map(token -> "'" + token.replaceAll("'", "\\\\'") + "'")
                                 .collect(Collectors.joining(","))
                + "]";
    }
}
