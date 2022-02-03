import knnClassification
import seaborn
import matplotlib.pyplot as plt
from sklearn import decomposition

temp_list = []


def listToNums(tar_list):
    for i in range(len(tar_list)):
        temp_list.append(ord(tar_list[i]))
    return temp_list


pca = decomposition.FactorAnalysis(n_components=2)
X = pca.fit_transform(knnClassification.test_BOW[:, :-1])
cmap = seaborn.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(X[:, 0], X[:, 1], c=listToNums(knnClassification.test_preds), s=100, cmap=cmap)
f.colorbar(points)
plt.show()
