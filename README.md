# Valutazione-CPV
L'obiettivo è valutare un sistema che genera automaticamente, a partire dall'oggetto di bandi di gara, i corrispondenti codici CPV (Common Procurement Vocabulary, sistema di classificazione unico per gli appalti pubblici volto a unificare i riferimenti utilizzati dalle amministrazioni e dagli enti apppaltanti per la descrizione dell'oggetto degli appalti) e della descrizione associata al codice.

Il vocabolario principale è composto da quattro livelli di gerarchia che rispettivamente vengono denominati:
- *divisioni*: utilizzano le prime due cifre del codice (XX000000-Y);
- *gruppi*: identificati dalle prime tre cifre del codice (XXX00000-Y);
- *classi*: sfruttano le prime quattro cifre del codice (XXXX0000-Y);
- *categorie*: formate dalle prime cinque cifre del codice (XXXXX000-Y).

La valutazione avviene impiegando un test set di 1.000.000 di elementi (si pone a confronto il risultato generato con quello corretto), e viene effettuata in questo modo:
- si calcola il numero di:
    - elementi generati con stesso codice e descrizione;
    - elementi generati con stesso codice, descrizione diversa;
    - elementi generati con stessa categoria;
    - elementi generati con stessa classe;
    - elementi generati con stesso gruppo;
    - elementi generati con stessa divisione.

Inoltre, si impiega la **metrica Bleu** per individuare la corrispondenza tra le descrizioni generate e quelle corrette. Quindi, viene calcolato:
- la media totale dei punteggi Bleu ottenuti;
- la media dei Bleu score di elementi con codice corrispondente ma descrizione diversa;
- la media dei Bleu score di elementi con categoria corrispondente;
- la media dei Bleu score di elementi con classe corrispondente;
- la media dei Bleu score di elementi con gruppo corrispondente;
- la media dei Bleu score di elementi con divisione corrispondente;
- la media dei Bleu score di elementi con nessuna corrispondenza.

## **Risultati**
I risultati della valutazione sono stati inseriti all'interno di un file Excel. Il file corrisponde al percorso: **Valutazione-CPV/AnalysisCPV/spreadsheet1.xlsx**.
I risultati ottenuti sono presentati nella seguente tabella:

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|372188 |1,0|
|Codice corrispondente|2403|0,466484497920533|
|Categoria corrispondente|76735|0,25729930592887|
|Classe corrispondente|115614|0,225988635255609|
|Gruppo corrispondente|109190|0,135904133495634|
|Divisione corrispondente|110555|0,167952746307877|
|Non corrispondenti|213315|0,0860599521618884|
|Tutti i test|1000000|0,454648514041545|

In seguito, sono stati ottenuti dei nuovi risultati impiegando il modello it5-base, come riportato nel foglio di calcolo Excel disponibile al percorso **Valutazione-CPV/AnalysisCPV/spreadsheet2.xlsx**. I risultati sono i seguenti:

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|359408 |1,0|
|Codice corrispondente|2210|0,485427223340654|
|Categoria corrispondente|73885|0,243696297543837|
|Classe corrispondente|114487|0,219889511329024|
|Gruppo corrispondente|114430|0,123508325003425|
|Divisione corrispondente|111538|0,163494735526761|
|Non corrispondenti|224042|0,0852636605159465|
|Tutti i test|1000000|0,438534472855048|

## **Valutazione: Lucene**

Successivamente, si è deciso di impiegare la libreria Java *Lucene*, la quale offre funzionalità di indicizzazione e di ricerca, per individuare, in corrispondenza di ogni esempio di test, la stringa corrispondente alla descrizione del CPV più simile.

Una volta realizzato il programma ottenente tali risultati, il quale offre in output documenti analoghi a quelli impiegati per le due valutazioni precedenti, questi sono stati valutati seguendo lo stesso criterio fin ora adottato.

Nello specifico, sono stati eseguiti i seguenti quattro test, caratterizzati da differenti configurazioni nella creazione dell'IndexWriter:
1. Analizzatore adottato StandardAnalyzer(), Similarità impiegata di default;
2. Analizzatore adottato StandardAnalyzer(), Similarità impiegata LMDirichletSimilarity();
3. Analizzatore adottato ItalianAnalyzer(), Similarità impiegata di default;
4. Analizzatore adottato ItalianAnalyzer(), Similarità impiegata LMDirichletSimilarity();

I quattro test hanno prodotto i seguenti risultati:

## *StandardAnalyzer*
Il file .json corrispondente è disponibile nel seguente percorso: **Valutazione-CPV/AnalysisCPV/cpv_StandardAnalyzer_generated.json**

Il file .xlsx corrispondente contenente i risultati è disponibile in: **Valutazione-CPV/AnalysisCPV/spreadsheetSA.xlsx**

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|104620 |1,0|
|Codice corrispondente|803|0,449656037289538|
|Categoria corrispondente|30692|0,227741300666648|
|Classe corrispondente|52246|0,153330286631564|
|Gruppo corrispondente|75096|0,130187556855609|
|Divisione corrispondente|104331|0,149883553123766|
|Non corrispondenti|632212|0,0548244470121678|
|Tutti i test|1000000|0,167233762294609|

## *StandardAnalyzer + LMDirichletSimilarity*
Il file .json corrispondente è disponibile nel seguente percorso: **Valutazione-CPV/AnalysisCPV/cpv_StandardAnalyzer_LMS_generated.json**

Il file .xlsx corrispondente contenente i risultati è disponibile in: **Valutazione-CPV/AnalysisCPV/spreadsheetSA_LMS.xlsx**

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|81599|1,0|
|Codice corrispondente|529|0,51271596927883|
|Categoria corrispondente|24578|0,191465872969654|
|Classe corrispondente|44128|0,128779279008883|
|Gruppo corrispondente|56648|0,11327414916708|
|Divisione corrispondente|76194|0,0997105378410726|
|Non corrispondenti|716324|0,0438965880978922|
|Tutti i test|1000000|0,129070141621589|

## *ItalianAnalyzer*
Il file .json corrispondente è disponibile nel seguente percorso: **Valutazione-CPV/AnalysisCPV/cpv_ItalianAnalyzer_generated.json**

Il file .xlsx corrispondente contenente i risultati è disponibile in: **Valutazione-CPV/AnalysisCPV/spreadsheetIA.xlsx**

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|113440|1,0|
|Codice corrispondente|872|0,512779651900972|
|Categoria corrispondente|30502|0,245211136537703|
|Classe corrispondente|53232|0,175571256356297|
|Gruppo corrispondente|90673|0,103956673617984|
|Divisione corrispondente|93574|0,156519903479547|
|Non corrispondenti|617707|0,0515140056038582|
|Tutti i test|1000000|0,175147430511434|

## *ItalianAnalyzer + LMDirichletSimilarity*
Il file .json corrispondente è disponibile nel seguente percorso: **Valutazione-CPV/AnalysisCPV/cpv_ItalianAnalyzer_LMS_generated.json**

Il file .xlsx corrispondente contenente i risultati è disponibile in: **Valutazione-CPV/AnalysisCPV/spreadsheetIA_LMS.xlsx**

| |Quantità |Media BLEU score |
|------------ |------------ |------------ |
|Completamente corrispondente|91564|1,0|
|Codice corrispondente|798|0,562483568573745|
|Categoria corrispondente|29322|0,17583373188281|
|Classe corrispondente|44105|0,123217909767146|
|Gruppo corrispondente|88935|0,0838203509635442|
|Divisione corrispondente|88524|0,115164788124562|
|Non corrispondenti|656752|0,046685475899141|
|Tutti i test|1000000|0,13946429248302|