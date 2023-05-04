import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;


public class LuceneModel {

    public static void main(String[] args) {

        Map<String, String> cpvMap = XmlUtility.xmlReader(new File("cpv_2008.xml"));

        List<CpvData> cpvData = CpvData.jsonReader(new File("TestJSON2.json"));

        try {

            FSDirectory fsdir = FSDirectory.open(new File("./resources/LuceneModel").toPath());

            Config config = Config.ITALIAN_ANALYZER_LMS;

            IndexWriter writer = new IndexWriter(fsdir, config.getIwc());


            // creazione dei documenti sui quali effettuare la ricerca
            for (Map.Entry<String, String> set : cpvMap.entrySet()) {

                Document doc = new Document();
                doc.add(new TextField("code", set.getKey(), Field.Store.YES));
                doc.add(new TextField("description", set.getValue(), Field.Store.YES));

                writer.addDocument(doc);

            }

            writer.close();


            IndexSearcher searcher = new IndexSearcher(DirectoryReader.open(fsdir));

            // !!!
            searcher.setSimilarity(config.getIwc().getSimilarity());

            QueryParser qp = new QueryParser("description", config.getIwc().getAnalyzer());

            BufferedWriter testWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream
                    (config.getTest())));

            BufferedWriter resultWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream
                    (config.getResult())));


            for (CpvData data : cpvData) {

                // la stringa query viene estratta e modificata per favorire il parsing e la ricerca
                String source = data.getSource().toLowerCase()
                        .replaceAll("[^a-z&&[^\\s]]", "");

                Query q = qp.parse(source);

                // scrittura del primo risultato in corrispondenza di ogni query (generated)
                TopDocs firstDoc = searcher.search(q, 1);
                ScoreDoc fDoc = firstDoc.scoreDocs[0];

                String[] generated = new String[1];
                generated[0] = CpvTest.toGenerated(searcher.doc(fDoc.doc).get("code"),
                        searcher.doc(fDoc.doc).get("description"));

                CpvTest test = new CpvTest(data.getSource(), data.getTarget(), generated);
                test.jsonWriter(testWriter);


                // scrittura dei primi 100 risultati generati in corrispondenza di ogni query
                TopDocs topdocs = searcher.search(q, 100);

                String[] topResults = new String[100];
                int i = 0;
                for (ScoreDoc sDoc : topdocs.scoreDocs) {

                    // devo inserire ogni stringa generata all'interno di un array
                    topResults[i] = CpvTest.toGenerated(searcher.doc(sDoc.doc).get("code"),
                            searcher.doc(sDoc.doc).get("description"));

                    i++;
                }

                CpvTest results = new CpvTest(data.getSource(), data.getTarget(), topResults);
                results.jsonWriter(resultWriter);

            }

            testWriter.close();
            resultWriter.close();

        } catch (IOException | ParseException ex) {
            Logger.getLogger(LuceneModel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}