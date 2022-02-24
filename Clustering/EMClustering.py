
import numpy as np
import DataBowTfidf
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt


target_arr = np.array(DataBowTfidf.tfidf_mat)
tsne = TSNE(n_components = 2)
coordinate = tsne.fit_transform(target_arr)
print(coordinate)

gmm = GaussianMixture(n_components=5)
gmm.fit(coordinate)
labels = gmm.predict(coordinate)
plt.scatter(coordinate[:, 0], coordinate[:, 1], c=labels, cmap='viridis');
gmm.predict_proba(coordinate)
plt.show()