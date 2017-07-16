"""
Python 2.x code for trying out different clustering algorithms for high-dimensional vectors
"""

import random

from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster.k_means_ import MiniBatchKMeans
# from sklearn.cluster.birch import Birch
# from sklearn.cluster.dbscan_ import DBSCAN
# from sklearn.cluster.mean_shift_ import MeanShift
# from sklearn.cluster.hierarchical import AgglomerativeClustering
# from sklearn.externals.joblib.memory import Memory
from sklearn.externals.joblib.memory import Memory

__author__ = 'aam'


def read_dict(voc=None):
    wv = {}
    count = 0

    with open("sbw_vectors_filtered.txt", 'r') as f:

        for line in f:
            if count % 10000 == 0:
                print count
            count += 1

            # format: word;number,number,number,...,number
            word, rest = line.split("\t")
            word = word.decode("utf-8")

            if voc is None or word in voc:
                rest = map(lambda x: float(x), rest.split(";"))
                wv[word] = rest
    return wv


w2v = read_dict()

print "Map size", len(w2v)

w2v_items = w2v.items()

random.shuffle(w2v_items)

keys = [w[0] for w in w2v_items]
vals = [w[1] for w in w2v_items]

clusterers = [
    # (KMeans(n_clusters=15, n_init=3, n_jobs=3, verbose=1), "kmeans15_init3"),
    # (KMeans(n_clusters=70, n_init=3, n_jobs=3, verbose=1), "kmeans70_init3"),
    # (KMeans(n_clusters=150, n_init=3, n_jobs=3, verbose=1), "kmeans150_init3"),
    # (KMeans(n_clusters=300, n_init=3, n_jobs=3, verbose=1), "kmeans300_init3")#,
    # (KMeans(n_clusters=500, n_init=3, n_jobs=3, verbose=1), "kmeans500_init3")#,
    # (MiniBatchKMeans(n_clusters=100, verbose=1, init_size=600), "MiniBatchKMeans100_init600"),
    # (MiniBatchKMeans(n_clusters=200, verbose=1, init_size=600), "MiniBatchKMeans200_init600"),
    # (MiniBatchKMeans(n_clusters=400, verbose=1, init_size=600), "MiniBatchKMeans400_init600"),
    # (MiniBatchKMeans(n_clusters=500, verbose=1, init_size=600), "MiniBatchKMeans500_init600"),
    # (MiniBatchKMeans(n_clusters=600, verbose=1, init_size=600), "MiniBatchKMeans600_init600")
    # (DBSCAN(), "DBSCAN_default")
    # (AgglomerativeClustering(n_clusters=15, linkage='ward'), "ward_hierarchial15"),
    # (AgglomerativeClustering(n_clusters=15, memory=Memory(cachedir="cache", verbose=1), linkage='average',
    #                          affinity='cosine'), "avg_hierarchial15_cosine"),
    # (AgglomerativeClustering(n_clusters=15, memory=Memory(cachedir="cache", verbose=1), linkage='average'),
    #  "avg_hierarchial15_euclidean"),
    # (AgglomerativeClustering(n_clusters=15, memory=Memory(cachedir="cache", verbose=1), linkage='complete'),
    #  "complete_hierarchial15_euclidean"),
    # (AgglomerativeClustering(n_clusters=15, memory=Memory(cachedir="cache", verbose=1), linkage='complete',
    #                          affinity='cosine'), "complete_hierarchial15_cosine"),
    # (AgglomerativeClustering(n_clusters=150, memory=Memory(cachedir="cache", verbose=1), linkage='average',
    #                          affinity='cosine'), "avg_hierarchial150_cosine"),
    # (AgglomerativeClustering(n_clusters=10, memory=Memory(cachedir="cache", verbose=1), linkage='average'),
    #  "avg_hierarchial10_euclidean"),
    # (AgglomerativeClustering(n_clusters=150, memory=Memory(cachedir="cache", verbose=1), linkage='complete'),
    #  "complete_hierarchial150_euclidean"),
    # (AgglomerativeClustering(n_clusters=150, memory=Memory(cachedir="cache", verbose=1), linkage='complete',
    #                          affinity='cosine'), "complete_hierarchial150_cosine"),
    # (MeanShift(), "meanshift_default"),
    # (DBSCAN(algorithm='kd_tree', eps=1.0), "DBSCAN_kdtree_eps1"),
    # (DBSCAN(algorithm='kd_tree', eps=5.0), "DBSCAN_kdtree_eps5"),
    # (Birch(n_clusters=15), "birch15"),
    # (Birch(n_clusters=70), "birch70"),
    # (MiniBatchKMeans(n_clusters=1000, verbose=1, init_size=12000), "MiniBatchKMeans1000_init12000"),
    (MiniBatchKMeans(n_clusters=50, verbose=1, init_size=500), "MiniBatchKMeans50_init500"),
]


def report(clrs):

    for clr, clr_sign in clrs:

        print clr_sign, ":", clr

        clusters = clr.fit(vals)
        labels = clusters.labels_

        with open("results/" + clr_sign + ".clusters", "wr+") as f:

            print "building clusters map"
            clusters_map = {}

            for id in xrange(len(labels)):
                if not labels[id] in clusters_map:
                    clusters_map[labels[id]] = [id]
                else:
                    clusters_map[labels[id]] += [id]

            print "writing stuff to file:", clr_sign

            for cluster_id, ids in clusters_map.iteritems():
                f.write("Cluster " + str(cluster_id) + "\n")
                for id in ids:
                    f.write(keys[id].encode("utf-8") + "\t")
                f.write("\n")


report(clusterers)