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