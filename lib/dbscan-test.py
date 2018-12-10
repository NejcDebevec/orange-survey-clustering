import unittest
import sklearn
import sklearn.datasets
import numpy as np
import dbscan
import pandas as pd
import matplotlib.pyplot as plt


class DBSCANTest(unittest.TestCase):
    def setUp(self):
        print("dela")
        # self.X = sklearn.datasets.load_iris().data
        # print(sklearn.datasets.load_iris().data)
        data = pd.read_csv("test_data.csv")

        data = data.drop("Unnamed: 0", axis=1)
        self.X = data.values
        # dbs = dbscan.DBSCAN(eps=0.4, min_samples=5)
        # clusters = dbs.fit_predict(self.X)

    def test_dbscan(self):
        # dbs = dbscan.DBSCAN(eps=1.4, min_samples=8)
        # # print(np.shape(self.X))
        # clusters = dbs.fit_predict(self.X)
        # print(np.unique(clusters))

        K_2 = dbscan.k_dist(self.X, '', 4)
        K_3 = dbscan.k_dist(self.X, '', 5)
        K_8 = dbscan.k_dist(self.X, '', 10)
        print(sorted(K_2, reverse=True))
        print(sorted(K_3, reverse=True))
        print(sorted(K_8, reverse=True))
        plt.plot(sorted(K_2, reverse=True))
        plt.plot(sorted(K_3, reverse=True))
        plt.plot(sorted(K_8, reverse=True))
        plt.legend(["k = 2", "k = 3", "k = 8"])
        plt.xlabel('Število primerov')
        plt.ylabel('Razdalja do k-tega soseda')
        plt.savefig('k-dist.pdf')
        plt.close()
    #     dbs = dbscan.DBSCAN(eps=0.4, min_samples=5)
    #     clusters = dbs.fit_predict(self.X)
    #     self.assertEqual(len(np.unique(clusters[clusters >= 0])), 2)
    #
    #     dbs = dbscan.DBSCAN(eps=0.2)
    #     clusters = dbs.fit_predict(self.X)
    #     self.assertGreater(len(np.unique(clusters[clusters >= 0])), 3)
    #
    #     self.assertGreater(np.sum(clusters < 0), 3)
    #
    # def test_kdist(self):
    #     d = dbscan.k_dist(self.X, metric="euclidean", k=4)
    #     self.assertGreater(min(d), -0.01)
    #     self.assertGreater(0.6, max(d))
    #     self.assertEqual(len(d), len(self.X))


if __name__ == "__main__":
    unittest.main(verbosity=2)
