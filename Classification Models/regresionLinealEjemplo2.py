'''
En el mundo de DataScience, Machine Learning y de Inteligencia Artificial,
Suele ser muy tipico usar python y R.
'''
import numpy as np   #Calculo numerico, operaciones matriciales, etc
import scipy as sc   #Libreria que hace lo mismo que numpy pero lo extiende a mayores herramientas cientificas
from scipy import linalg
import sklearn as sk #Libreria de machine learning que tambien incluye datasets
from matplotlib import pyplot as plt #Visualizacion grafica de datos
from sklearn.datasets import load_boston

'''
Formula para minimizar el error cuadratico medio (tecnica de minimos cuadrados)

Betha = ((X * X{traspuesta})**{-1}) * X{transpuesta} * Y
'''


boston = load_boston()
#print boston.DESCR podemos ver la descripcion del dataset

#Matriz de entrada
X = np.array(boston.data[:, 5]) #La columna que nos indica el numero medio de habitaciones
y = np.array(boston .target) #Valor medio de las casas

plt.scatter(X,y, alpha = 0.3)

#Anadiendo una comlumna de unos para termino independiente
X = np.array([np.ones(506), X]).T
#Con el signo arroba @ se consigue la multiplicacion matricial (Formula mencionada arriba)
#B = np.linalg.inv(X.T @  X) @ X.T @ y
B = np.dot(np.linalg.inv(np.dot(X.T, X)),  np.dot(X.T, y))  #Al imprimir B, nos da [-34.67062078   9.10210898], en donde 9.10 es la pendiente y -34.67 es b en donde corta al eje Y cuando x = 0

plt.plot([4,9],[B[0] + B[1] * 4, B[0] + B[1] * 9], c = "red")
#Aqui podemos ver graficada nuestra nube de datos
plt.xlabel('No. Habitaciones Medio')
plt.ylabel('Precio Casas')
plt.show()
