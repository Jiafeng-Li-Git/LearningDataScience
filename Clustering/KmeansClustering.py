
import pandas as pd
import numpy as np
import DataBowTfidf
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


target_arr = np.array(DataBowTfidf.tfidf_mat)
codinate = TSNE(n_components=2).fit_transform(target_arr)
kmeans = KMeans(n_clusters=5, random_state=0).fit(target_arr)
df_tsne = pd.DataFrame({'dimension1': codinate[:, 0], 'dimension2': codinate[:, 1], 'label': kmeans.labels_})
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_tsne, hue='label', x='dimension1', y='dimension2', s=80)
plt.show()