from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DIV = 2
GROUP = 3
CLASS = 4
CAT = 5
COD = 10
DESCR = 13


def has_same_string(reference, generated):
    if reference == generated:
        return True


def has_same_code(reference, generated):
    if reference[:COD] == generated[:COD]:
        return True


def has_same_cat(reference, generated):
    if reference[:CAT] == generated[:CAT]:
        return True


def has_same_class(reference, generated):
    if reference[:CLASS] == generated[:CLASS]:
        return True


def has_same_group(reference, generated):
    if reference[:GROUP] == generated[:GROUP]:
        return True


def has_same_div(reference, generated):
    if reference[:DIV] == generated[:DIV]:
        return True
