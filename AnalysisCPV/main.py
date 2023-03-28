import json
import cpvmanager

from cpvmanager import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

noticeList = []

sameCode = 0
sameString = 0
sameCat = 0
sameClass = 0
sameGroup = 0
sameDiv = 0
notMatching = 0

bs = 0
bsSameCode = 0
bsSameCat = 0
bsSameClass = 0
bsSameGroup = 0
bsSameDiv = 0
bsNotMatching = 0

# SmoothingFunction() consente di evitare di essere penalizzati dall'assenza di corrispondenze di n-gram tra frase
# target e frase generata
chencherry = SmoothingFunction()

with open('cpv_5M_generated.json') as file:
    for objJSON in file:
        notice = json.loads(objJSON)
        noticeList.append(notice)

for element in noticeList:
    ref = element["target"]
    gen = element["generated"][0]

    # calcoliamo il punteggio considerando unicamente 1-gram e 2-gram per evitare di avere valori troppo bassi,
    # considerata la dimensione delle stringhe
    bleuScore = sentence_bleu([ref[cpvmanager.DESCR:].split()], gen[cpvmanager.DESCR:].split(),
                              weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)

    bs += bleuScore

    if has_same_string(ref, gen):
        sameString += 1
    elif has_same_code(ref, gen):
        sameCode += 1
        bsSameCode += bleuScore
    elif has_same_cat(ref, gen):
        sameCat += 1
        bsSameCat += bleuScore
    elif has_same_class(ref, gen):
        sameClass += 1
        bsSameClass += bleuScore
    elif has_same_group(ref, gen):
        sameGroup += 1
        bsSameGroup += bleuScore
    elif has_same_div(ref, gen):
        sameDiv += 1
        bsSameDiv += bleuScore
    else:
        notMatching += 1
        bsNotMatching += bleuScore

# calcolo della media del Bleu score
dim = len(noticeList)
bs = bs / dim
bsSameCode = bsSameCode / sameCode
bsSameCat = bsSameCat / sameCat
bsSameDiv = bsSameDiv / sameDiv
bsSameGroup = bsSameGroup / sameGroup
bsSameClass = bsSameClass / sameClass
bsNotMatching = bsNotMatching / notMatching

print("numero test = {}".format(dim))
print("I test hanno riportato i seguenti risultati:")
print("- presentanti codice e descrizione corrispondente = {}".format(sameString))
print("- presentanti codice corrispondente = {}".format(sameCode))
print("- presentanti categoria corrispondente = {}".format(sameCat))
print("- presentanti classe corrispondente = {}".format(sameClass))
print('- presentanti gruppo corrispondente = {}'.format(sameGroup))
print('- presentanti divisione corrispondente = {}'.format(sameDiv))
print('- In nessun modo corrispondenti = {}'.format(notMatching))
print("- Media Bleu score totale = {}".format(bs))
print("- Media Bleu score di elementi con stesso codice ma descrizione diversa = {}".format(bsSameCode))
print("- Media Bleu score di elementi con stessa categoria = {}".format(bsSameCat))
print("- Media Bleu score di elementi con stessa classe = {}".format(bsSameClass))
print("- Media Bleu score di elementi con stesso gruppo = {}".format(bsSameGroup))
print("- Media Bleu score di elementi con stessa divisione = {}".format(bsSameDiv))
print("- Media Bleu score di elementi non corrispondenti = {}".format(bsNotMatching))
