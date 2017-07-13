# coding: utf8
import os


from pattern.text.es import parsetree
import re

# stopwords = set(
#     "un una unas unos uno sobre todo también tras otro algún alguno alguna algunos algunas ser es soy eres somos
# sois estoy esta estamos estais estan como en para atras porque por qué estado estaba ante antes siendo ambos pero por poder puede puedo podemos podeis pueden fui fue fuimos fueron hacer hago hace hacemos haceis hacen cada fin incluso primero desde conseguir consigo consigue consigues conseguimos consiguen ir voy va vamos vais van vaya gueno ha tener tengo tiene tenemos teneis tienen el la lo las los su aqui mio tuyo ellos ellas nos nosotros vosotros vosotras si dentro solo solamente saber sabes sabe sabemos sabeis saben ultimo largo bastante haces muchos aquellos aquellas sus entonces tiempo verdad verdadero verdadera cierto ciertos cierta ciertas intentar intento intenta intentas intentamos intentais intentan dos bajo arriba encima usar uso usas usa usamos usais usan emplear empleo empleas emplean ampleamos empleais valor muy era eras eramos eran modo bien cual cuando donde mientras quien con entre sin trabajo trabajar trabajas trabaja trabajamos trabajais trabajan podria podrias podriamos podrian podriais yo aquel".split())

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

from gensim import corpora
dictionary = corpora.Dictionary(prepared_texts)

print("Dictionary built!", dictionary)

corpus = [dictionary.doc2bow(text) for text in prepared_texts]

print("Corpus built!")

k = 5

import gensim

lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
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
    print("\n".join([dictionary[term] + "\t" + str(prob) for (term, prob) in lda.get_topic_terms(i, topn=15)]))
    print()
