import numpy as np
from  matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.datasets import load_digits


digits = load_digits()
data = digits.data

data = 255-data

np.random.seed(1)

n = 10

kmeans = KMeans(n_clusters=n,init='random')
kmeans.fit(data)
Z = kmeans.predict(data)

for i in range(0,n):

    fila = np.where(Z==i)[0] # filas en Z donde estan las imagenes de cada cluster
    num = fila.shape[0]      # numero imagenes de cada cluster
    r = np.floor(num/10.)    # numero de filas menos 1 en figura de salida

    print("cluster "+str(i))
    print(str(num)+" elementos")

    plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        imagen = data[fila[k], ]
        imagen = imagen.reshape(8, 8)
        plt.imshow(imagen, cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()
