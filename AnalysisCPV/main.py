import json
import cpvmanager

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

noticeList = []

sameCode = 0
sameString = 0
sameCat = 0
sameClass = 0
sameGroup = 0
sameDiv = 0
s = 0

# SmoothingFunction() consente di evitare di essere penalizzati dall'assenza di corrispondenze relativamente a un
# n-gram tra frase target e frase generata
chencherry = SmoothingFunction()

# elimino l'eventuale contenuto del file bleufile.txt
# with open("bleufile.txt", 'r+') as file:
#     file.truncate(0)


with open('cpv_5M_generated.json') as file:
    for objJSON in file:
        notice = json.loads(objJSON)
        noticeList.append(notice)

# with open('bleufile.txt', 'a') as f:
#     f.write(str(sentence_bleu(ref, gen, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)))
#     f.write('\n')
# f.close()

for element in noticeList:
    ref = element["target"]
    gen = element["generated"][0]

    s += sentence_bleu([ref[cpvmanager.DESCR:].split()], gen[cpvmanager.DESCR:].split(),
                       weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)

    if ref == gen:
        sameString += 1
    if ref[:cpvmanager.COD] == gen[:cpvmanager.COD]:
        sameCode += 1
    if ref[:cpvmanager.CAT] == gen[:cpvmanager.CAT]:
        sameCat += 1
    if ref[:cpvmanager.CLASS] == gen[:cpvmanager.CLASS]:
        sameClass += 1
    if ref[:cpvmanager.GROUP] == gen[:cpvmanager.GROUP]:
        sameGroup += 1
    if ref[:cpvmanager.DIV] == gen[:cpvmanager.DIV]:
        sameDiv += 1

avg = s / (len(noticeList))

print(len(noticeList))
print(sameString)
print(sameCode)
print(sameCat)
print(sameClass)
print(sameGroup)
print(sameDiv)
print(avg)

# gen1 = noticeList[0]["generated"][0].split()
# print(sentence_bleu(ref1, gen1, weights=(1, 0, 0, 0)))

# ref2 = [noticeList[2]["target"].split()]
# gen2 = noticeList[2]["generated"][0].split()
# print(sentence_bleu(ref2, gen2, smoothing_function=chencherry.method1))
# print(sentence_bleu(ref2, target2, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))
# print(sentence_bleu(ref2, target2, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
