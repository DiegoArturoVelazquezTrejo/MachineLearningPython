from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
#xs = np.array([1,2,3,4,5,6], dtype = np.float64)
#ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def create_dataset_example(hm, variance, step = 2, correlation = False):
    #La varianza es que tan lejanos o cercanos estaran los puntos de la linea
    #Mientras menor sea, estaran pegados a la linea de regresion, por lo que el
    #coeficiente de determionacion sera proximo a 1
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y) #This will give us data
        if(correlation and correlation == 'pos'):
            val += step
        elif(correlation and correlation == 'neg'):
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)


#Calculating the slope (pendiente) and the y interception
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
        ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m * mean(xs)
    return m, b

#Calculating squared error
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

#Calculating coeficient of determination
def coeficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_y_mean)

xs, ys = create_dataset_example(40, 80, 2, correlation = 'pos')

m, b = best_fit_slope_and_intercept(xs, ys) #Aqui con la funcion generamos dataset con valores aleatorios para probar el algoritmo

#Creating the line that fits the data in the graphic

regression_line = [ (m * x) + b for x in xs]

#Making prediction based on the data and on the model
predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coeficient_of_determination(ys, regression_line)
print r_squared #We want this value to be zero

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'green')
plt.plot(xs, regression_line)
plt.show()
