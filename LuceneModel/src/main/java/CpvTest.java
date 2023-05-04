import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class CpvTest {
    private final String source;
    private final String target;
    private final String[] generated;


    public CpvTest(String source, String target, String[] generated) {
        this.source = source;
        this.target = target;
        this.generated = generated;
    }


    public String getSource() {
        return source;
    }


    public String getTarget() {
        return target;
    }


    public String[] getGenerated() {
        return generated;
    }


    public static String toGenerated(String code, String description) {
        String string = ("\"" + code + " - " + description.toLowerCase() + "\"");
        return string;
    }


    public void jsonWriter(BufferedWriter writer) {
        try {
            writer.write(this.toString());
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static List<CpvTest> jsonReader(File file) {
        List<CpvTest> cpvTest = new ArrayList();
        try {

            Scanner scanner = new Scanner(file);

            while (scanner.hasNextLine()) {
                String jsonString = scanner.nextLine();

                GsonBuilder builder = new GsonBuilder();
                builder.setPrettyPrinting();

                Gson gson = builder.create();
                CpvTest cpvElement = gson.fromJson(jsonString, CpvTest.class);

                cpvTest.add(cpvElement);
            }

            scanner.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return cpvTest;
    }


    private String generatedToString() {

        StringBuilder sb = new StringBuilder();

        int i = 0;
        for (String s : this.generated) {
            sb.append(s);
            i++;

            if (this.generated.length != i) {
                sb.append(", ");
            }
        }

        return sb.toString();
    }


    public String toString() {
        String string = ("{\"source\": \"" + source + "\", " + "\"target\": \"" + target + "\", " + "\"generated\": ["
                + this.generatedToString() + "]}");
        return string;
    }

}
