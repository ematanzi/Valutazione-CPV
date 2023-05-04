import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class CpvData {
    private final String source;
    private final String target;


    public CpvData(String source, String target) {
        this.source = source;
        this.target = target;
    }


    public String getSource() {
        return source;
    }


    public String getTarget() {
        return target;
    }


    public static List<CpvData> jsonReader(File file) {
        List<CpvData> cpvData = new ArrayList();

        try {

            BufferedReader br = new BufferedReader(new FileReader(file));

            while (br.readLine() != null) {
                String jsonString = br.readLine();

                GsonBuilder builder = new GsonBuilder();
                builder.setPrettyPrinting();

                Gson gson = builder.create();
                CpvData cpvElement = gson.fromJson(jsonString, CpvData.class);

                cpvData.add(cpvElement);
            }

            br.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return cpvData;
    }

    public void jsonWriter(BufferedWriter writer) {

        try {

            writer.write(this.toString());
            writer.newLine();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    @Override
    public String toString() {
        String string = ("{\"source\": \"" + source + "\", " + "\"target\": \"" + target + "\"}");
        return string;
    }
}
