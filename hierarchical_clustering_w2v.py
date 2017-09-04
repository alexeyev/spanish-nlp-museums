from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from collections import defaultdict

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

from spatial_clustering_w2v import read_dict

wv_dict = read_dict()

words = []
vectors = []

for word in wv_dict:
    # print word, wv_dict[word]
    words.append(word)
    vectors.append(wv_dict[word])

X = np.array(vectors)

Z = linkage(X, method='average', metric='cosine')
# Z = linkage(X, 'centroid', metric='euclidean')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# the cophenetic distances between each observation in the hierarchical clustering defined by the linkage
c, coph_dists = cophenet(Z, pdist(X))

print c

# calculate full dendrogram
# plt.figure(figsize=(25, 10))
# plt.title('Clustering Dendrogram')
#
# plt.xlabel('index')
# plt.ylabel('distance')
#
# dendrogram(
#     Z,
#     truncate_mode='lastp',  # show only the last p merged clusters
#     p=100,  # show only the last p merged clusters
#     leaf_rotation=90.,
#     leaf_font_size=12.,
#     show_contracted=True,  # to get a distribution impression in truncated branches
# )
# plt.show()

# k = 50
# clusters = fcluster(Z, k, criterion='maxclust')

clusters = None

for t in np.linspace(0.7, 0.72, num=5):
    clusters = fcluster(Z, t=t, criterion="distance")
    print t, len(set(list(clusters)))

cluster_dict = defaultdict(lambda: [])

for cl, name in zip(clusters, words):
    cluster_dict[cl].append(name)

for k in cluster_dict:
    print k, len(cluster_dict[k])
    print " ".join(cluster_dict[k])
