import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;

public class XmlUtility {

    public static Map<String, String> xmlReader(File file) {

        Map<String, String> cpvDictionary = new HashMap<>();

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

        try {

            DocumentBuilder builder = factory.newDocumentBuilder();

            Document document = builder.parse(file);

            document.getDocumentElement().normalize();

            NodeList cpvList = document.getElementsByTagName("CPV");

            for (int i = 0; i < cpvList.getLength(); i++) {
                Node cpv = cpvList.item(i);

                if (cpv.getNodeType() == Node.ELEMENT_NODE) {
                    Element cpvElement = (Element) cpv;

                    NodeList cpvDetails = cpv.getChildNodes();

                    for (int j = 0; j < cpvDetails.getLength(); j++) {

                        Node detail = cpvDetails.item(j);

                        if (detail.getNodeType() == Node.ELEMENT_NODE) {
                            Element detailElement = (Element) detail;

                            if (detailElement.getAttribute("LANG").equals("IT")) {

                                cpvDictionary.put(cpvElement.getAttribute("CODE"), detailElement.getTextContent());

                            }
                        }
                    }
                }
            }

        } catch (ParserConfigurationException | IOException | SAXException e) {
            e.printStackTrace();
        }

        return cpvDictionary;
    }

}
