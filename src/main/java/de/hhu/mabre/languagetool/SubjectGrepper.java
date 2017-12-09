package de.hhu.mabre.languagetool;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class SubjectGrepper {
    static List<String> grep(Reader fileReader, String... subjects) throws IOException {
        LinkedList<String> results = new LinkedList<>();
        BufferedReader reader = new BufferedReader(fileReader);
        String line;
        while((line = reader.readLine()) != null) {
            if(Arrays.stream(subjects).filter(line::contains).count() > 0) {
                results.add(line);
            }
        }
        return results;
    }
}
