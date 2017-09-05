# coding: utf8
import os
import re

from gensim import corpora
from pattern.text.es import parsetree

stopwords = set(map(lambda x: x.strip(), open("stopwords_es.txt", "r+").read().split("\n")))

big_texts = [open("data/es/" + f).read() for f in os.listdir("data/es/")]

segments = []

for bt in big_texts:

    lines = [s for s in bt.split("\n")]

    segments_t = [""]

    for line in lines:
        if line.strip():
            segments_t[len(segments_t) - 1] += line
        else:
            if segments_t[len(segments_t) - 1].strip():
                segments_t.append("")

    segments.extend(segments_t)

print "A total of", len(segments), "segments"


def split_text(txt):
    return re.split("\W+", txt.strip().lower().replace("\n", " "))


def lemmatize(txt):
    # нормализация + оставляем существительные
    return [w.lemma for s in parsetree(txt, lemmata=True).sentences for w in s.words if w.tag.startswith("NN")]


print "Parsing and filtering..."

prepared_texts = []

for text in segments:
    # нормализуем, оставляем существительные, выкидываем цифры, отбрасываем стоп-слова и короткие слова
    lemmatized = [word for word in lemmatize(text) if
                  not word in stopwords and not re.search("\d", word) and not len(word) < 4]
    prepared_texts.append(lemmatized)

dictionary = corpora.Dictionary(prepared_texts)

print("Dictionary built!", dictionary)

corpus = [dictionary.doc2bow(text) for text in prepared_texts]

print("Corpus built!")

k = 6

import gensim

ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           update_every=1,
                                           chunksize=70,
                                           passes=10,
                                           # iterations=1000,
                                           alpha='auto',
                                           eval_every=2,
                                           random_state=55
                                           )
print("LDA trained")

for i in range(k):
    print("Topic", i)
    print()
    print("\n".join([dictionary[term].encode("utf-8") + "\t" + str(prob) for (term, prob) in
                     ldamodel.get_topic_terms(i, topn=10)]))
    print()

with open("segmented_tm_topics" + str(k) + ".csv", "w") as wf:
    wf.write("topic_id,term,prob\n")
    for i in range(k):
        wf.write(
            "\n".join([str(i) + "," + dictionary[term].encode("utf-8") + "," + str(prob) for (term, prob) in
                       ldamodel.get_topic_terms(i, topn=2000)]))
        wf.write("\n\n")
