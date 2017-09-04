from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

from spatial_clustering_w2v import read_dict

wv_dict = read_dict()

words = []
vectors =[]

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
plt.figure(figsize=(25, 10))
plt.title('Clustering Dendrogram')

plt.xlabel('index')
plt.ylabel('distance')

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=50,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()