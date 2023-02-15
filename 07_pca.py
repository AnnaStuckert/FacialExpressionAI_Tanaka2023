"""
@author: tanakayudai
Multivariate analysis with principal component analysis
"""

import csv
import numpy as np
import pandas as pd
from pandas import plotting 
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

# Read csv file
df = pd.read_csv('*****')
# z-score normalization
dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
# Perform principal component analysis
pca = PCA()
pca.fit(dfs)
# Principal component score
feature = pca.transform(dfs)    
s2pca = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))])    
# Eigenvalues of PCA
pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
# Eigenvectors of PCA
pd.DataFrame(pca.components_, columns=dfs.columns[0:], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])

# 3D Scatter Plot
n1,p1,t1=feature[:, 0][0:3099], feature[:, 0][3099:6198], feature[:, 0][6198:9297]
n2,p2,t2=feature[:, 1][0:3099], feature[:, 1][3099:6198], feature[:, 1][6198:9297]
n3,p3,t3=feature[:, 2][0:3099], feature[:, 2][3099:6198], feature[:, 2][6198:9297]
fig = plt.figure(figsize = (8, 8))
# Add and set 3D axes
ax = fig.add_subplot(111, projection='3d')
ax.set_title("", size = 20)
ax.set_xlabel("PC1", size = 14, color = "black")
ax.set_ylabel("PC2", size = 14, color = "black")
ax.set_zlabel("PC3", size = 14, color = "black")
ax.set_xticks([-5.0, 0.0, 5.0])
ax.set_yticks([-5.0, 0.0, 5.0])
ax.set_zticks([-5.0, 0.0, 5.0])
# Draw a graph
ax.scatter(n1, n2, n3, s=40, c="black", alpha=0.1, label="Neutral") 
ax.scatter(p1, p2, p3, s=40, c="blue", alpha=0.1, label="Painful") 
ax.scatter(t1, t2, t3, s=40, c="orange",alpha=0.1, label="Tickling") 
ax.view_init(elev=30, azim=-35)

plt.show()
