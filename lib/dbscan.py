import sklearn.neighbors as sk
import numpy
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


# Definiram class DBSCAN
class DBSCAN:
    # Konstruktor za class DBSCAN v katerem sprejemo dva argumenta, eps in min_samples
    def __init__(self, eps=0.1, min_samples=4):
        self.eps = eps
        self.minPts = min_samples

    # Metoda v kateri izvedemo grućanje
    def fit_predict(self, data):
        self.data = data
        c = -1
        labels = [-4] * len(self.data)  # Nastavimo vektor skupin za vsako točko
        tree = sk.KDTree(self.data, leaf_size=2)  # zgradimo drevo, s katerim si bomo pomagali
                                                    # pri iskanju najbližjih točk

        for nu, point in enumerate(self.data):  # Sprehodimo se skozi vse točke
            if labels[nu] < -1:  # Če točka še ni obiskana
                n = tree.query_radius([point], r=self.eps)[0].tolist()  # poiščemo vse točke v soseščini
                if len(n) < self.minPts:  # če jih je manj kot je potrebno da tvorijo soseščino, točko označim kot šum
                    labels[nu] = -1
                else:  # če ne, je najden nov cluster
                    c += 1
                    labels[nu] = c
                    for p in n:  # sprehodimo se skozi vse točke v soseščini
                        if labels[p] == -1:  # če je bila ta točka do sedaj šum, jo prilagodimo novi skupini
                            labels[p] = c
                        if labels[p] < -1:  # če točka še ni bila obiskana, jo dodamo skupini in pošiščemo vse točke
                                            #  v njeni soseki
                            labels[p] = c
                            neighbourPts_2 = tree.query_radius([self.data[p]], r=self.eps)[0].tolist()
                            if len(neighbourPts_2) >= self.minPts:  # če je točk več kot jih mora bit za tvorjenje jedra
                                                                    # dodamo vse te točke v skupne točke soseščine
                                for k in neighbourPts_2:
                                    if k not in n:
                                        n.append(k)

        return numpy.array(labels)  # vrnemo vektor skupin

# funkcije v kateri dobimo za vsako točko k-tega najbolj odaljenega soseda
def k_dist(X, metric, k=3):
        data = []
        tree = sk.KDTree(X, leaf_size=30)
        for n,point in enumerate(X):
            dist, ind = tree.query([point], k=k)
            data.append(dist[0].tolist()[k-1])
        return data

# centers = [[1, 1], [-1, -1], [1, -1]]
# data, clust = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
#                             random_state=0)

# db = DBSCAN(0.2, 5)
# labels = db.fit_predict(data)
#
#
# K_2 = k_dist(data,'', 2)
# K_3 = k_dist(data,'', 3)
# K_8 = k_dist(data,'', 8)
#
#
# plt.plot(sorted(K_2, reverse=True))
# plt.plot(sorted(K_3, reverse=True))
# plt.plot(sorted(K_8, reverse=True))
# plt.legend(["k = 2", "k = 3", "k = 8"])
# plt.xlabel('Število primerov')
# plt.ylabel('Razdalja do k-tega soseda')
# plt.savefig('k-dist.pdf')
# plt.close()

# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
# unique_labels = set(labels)
#
# colors = [plt.cm.Spectral(each)
#           for each in numpy.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     vek = [True if a==k else False for a in labels]  # zgradimo vektor z true vrednostmi na ideksi kjer so elementi
#                                                         # v tej skupini
#     if k == -1:
#         # nastavimo črno barvo za šume
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = data[vek]  # narišemo točke
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
# plt.title('Stevilo najdenih skupin: %d' % n_clusters_)
#
# plt.savefig("cluster.pdf")
