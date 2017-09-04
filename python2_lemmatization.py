# coding: utf8
import os

from pattern.text.es import parsetree
import re
from gensim import corpora

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
    prepared_texts.append(lemmatized)


dictionary = corpora.Dictionary(prepared_texts)

print("Dictionary built!", dictionary)

corpus = [dictionary.doc2bow(text) for text in prepared_texts]

print("Corpus built!")

k = 5

import gensim

ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                      id2word=dictionary,
                                      num_topics=k,
                                      update_every=1,
                                      chunksize=100,
                                      passes=1)
print("LDA trained")

lsi = gensim.models.lsimodel.LsiModel(corpus=corpus,
                                      id2word=dictionary,
                                      num_topics=k)

print("LSI trained")

# for LDA

for i in range(k):
    print("Topic", i)
    print()
    print("\n".join([dictionary[term].encode("utf-8") + "\t" + str(prob) for (term, prob) in ldamodel.get_topic_terms(i, topn=100000)]))
    print()

with open("dasha-please-dont-send-screenshots-anymore-use-gists-or-whatever-because-i-cant-copy-and-paste.csv", "w") as wf:

    for i in range(k):
        wf.write(
            "\n".join([dictionary[term].encode("utf-8") + "," + str(prob) for (term, prob) in ldamodel.get_topic_terms(i, topn=100000)]))
        wf.write("\n")
