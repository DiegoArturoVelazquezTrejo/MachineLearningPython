from matplotlib import pyplot as plt
from matplotlib import style


#Este algoritmo trabaja de igual forma con las distancias
#Eucledianas y calcula la media de los centroides para redefinir un nuevo centroide que sera
#El cluster.

style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [1.5, 1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])

#plt.scatter(X[:,0],X[:,1],s = 150,c='b')
#plt.show()

clf = KMeans(n_clusters = 3)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10*["g.","r.","c.","b.","k."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)

plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 150, linewidth = 5)
plt.show()
