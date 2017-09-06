# coding:utf-8
import numpy as np

import os
import re

from pattern.text.es import parsetree

# переводы
es_words = [word.strip().strip(".") for word in open("words_only.txt", "r")]
en_words = [word.strip().lower() for word in open("en_gtranslate_words.txt", "r")]

es2en = {es: en for es, en in zip(es_words, en_words)}
print es2en
print es2en["trabajas"]

print(len(es_words), len(en_words))

with open("i-had-to-code-for-that.csv", "w+") as wf:
    wf.write("es,en\n")
    for i in range(len(es_words)):
        wf.write(es_words[i].replace(",", " ") + "," + en_words[i].replace(",", " ").replace(".", "") + "\n")

quit()

# мусорные слова
stopwords = set(map(lambda x: x.strip(), open("stopwords_es.txt", "r+").read().split("\n")))

# читаем наши тексты на испанском
texts = [open("data/es/" + f).read() for f in os.listdir("data/es/")]


def split_text(txt):
    return re.split("\W+", txt.strip().lower().replace("\n", " "))


def lemmatize(txt):
    print(parsetree(txt, lemmata=False).sentences)
    return [w.lemma for s in parsetree(txt, lemmata=True).sentences for w in s.words]


# лемматизируем тексты
# список списков слов
prepared_texts = []
prepared_words = set()

for text in texts:
    lemmatized = [word.lower() for word in lemmatize(text) if
                  not word in stopwords and not re.search("\d", word) and not len(word) < 4]
    prepared_texts.extend(lemmatized)
    prepared_words.update(lemmatized)

prepared_texts = set(prepared_texts)

print(len(prepared_texts))
print(prepared_texts)
print()
print(len(prepared_words))
print("WORDS", prepared_words)

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

    if word in prepared_words:
        # print(word, " is in ")
        w2v[word] = vector
    else:
        pass
        # print(word, "is not in dict")

with open("sbw_vectors_filtered.txt", "w+") as wf:
    for key in w2v:
        wf.write(key + "(" + es2en[key] + ")\t" + ";".join([str(i) for i in w2v[key].tolist()]) + "\n")
