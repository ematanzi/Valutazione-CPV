import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.it.ItalianAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMSimilarity;

import java.io.File;


public class Config {

    private final File test;
    private final File result;
    private final IndexWriterConfig iwc;

    private Config(File test, File result, Analyzer analyzer) {
        this.test = test;
        this.result = result;
        this.iwc = new IndexWriterConfig(analyzer);
    }

    private Config(File test, File result, Analyzer analyzer, LMSimilarity sim) {
        this.test = test;
        this.result = result;
        this.iwc = new IndexWriterConfig(analyzer).setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                .setSimilarity(sim);
    }


    public static Config STANDARD_ANALYZER = new Config(
            new File("cpv_StandardAnalyzer_generated.json"),
            new File("cpv_firstResults_SA.json"),
            new StandardAnalyzer());

    public static Config STANDARD_ANALYZER_LMS = new Config(
            new File("cpv_StandardAnalyzer_LMS_generated.json"),
            new File("cpv_firstResults_SA_LMS.json"),
            new StandardAnalyzer(),
            new LMDirichletSimilarity(new LMSimilarity.DefaultCollectionModel(), 2000));

    public static Config ITALIAN_ANALYZER = new Config(
            new File("cpv_ItalianAnalyzer_generated.json"),
            new File("cpv_firstResults_IA.json"),
            new ItalianAnalyzer());

    public static Config ITALIAN_ANALYZER_LMS = new Config(
            new File("cpv_ItalianAnalyzer_LMS_generated.json"),
            new File("cpv_firstResults_IA_LMS.json"),
            new ItalianAnalyzer(),
            new LMDirichletSimilarity(new LMSimilarity.DefaultCollectionModel(), 2000));


    public File getTest() {
        return test;
    }

    public File getResult() {
        return result;
    }

    public IndexWriterConfig getIwc() {
        return iwc;
    }


    @Override
    public String toString() {
        return "Config{" +
                "test=" + test +
                ", result=" + result +
                ", iwc=" + iwc +
                '}';
    }
}
