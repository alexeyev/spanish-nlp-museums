import numpy as np

import os
import re

from pattern.text.es import parsetree

stopwords = set(map(lambda x: x.strip(), open("stopwords_es.txt", "r+").read().split("\n")))

texts = [open("data/es/" + f).read() for f in os.listdir("data/es/")]


def split_text(txt):
    return re.split("\W+", txt.strip().lower().replace("\n", " "))


def lemmatize(txt):
    return [w.lemma for s in parsetree(txt, lemmata=True).sentences for w in s.words]


prepared_texts = []

for text in texts:
    lemmatized = [word for word in lemmatize(text) if
                  not word in stopwords and not re.search("\d", word) and not len(word) < 4]
    prepared_texts.extend(lemmatized)


prepared_texts = set(prepared_texts)

print(len(prepared_texts))
print(prepared_texts)


# reading only word2vecs of interest

firstline = True
w2v = {}

for line in open("w2v/SBW-vectors-300-min5.txt", "r+"):
    if firstline:
        s = line.split(" ")
        count = long(s[0])
        dim = int(s[1])
        firstline = False
        continue

    s = line.strip().split(" ")
    word = s[0]
    vector = np.array(list(map(lambda x: float(x), s[1:])))
    w2v[word] = vector