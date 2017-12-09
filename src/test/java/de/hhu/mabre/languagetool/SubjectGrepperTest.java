package de.hhu.mabre.languagetool;

import junit.framework.TestCase;

import java.io.StringReader;
import java.util.List;

import static de.hhu.mabre.languagetool.SubjectGrepper.grep;

public class SubjectGrepperTest extends TestCase {
    public void testGrepSingleSubject() throws Exception {
        List<String> results = grep(new StringReader("foo bar\nlorem ipsum\nblabfoobdfj\nasdf"), "foo");
        assertEquals(2, results.size());
        assertEquals("foo bar", results.get(0));
    }

    public void testGrepTwoSubjects() throws Exception {
        List<String> results = grep(new StringReader("foo bar\nlorem ipsum\nblabfoobdfj\nasdf"), "foo", "l");
        assertEquals(3, results.size());
        assertEquals("foo bar", results.get(0));
    }
}