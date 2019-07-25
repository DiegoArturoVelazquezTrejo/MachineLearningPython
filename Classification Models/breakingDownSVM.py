#Breaking down the SVM
#Turorial 23
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')


# STUDY CONVEX OPTIMIZATION!!!!!! FOR MACHINE LEARNING ALGORITHMS


class support_vector_machine:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'} #The class which has 1 got the red color in the graph, and the -1 will be blue
        if (self.visualization):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    #Train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b]  }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        #Support vectors yi(xi.w + b) = 1
        # 1.01

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #Point of expense
                      self.max_feature_value * 0.001,
                      self.max_feature_value * 0.0001]

        #Extremely expensive
        b_range_multiple = 5
        #
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            #We can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value * b_range_multiple), self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_opcion = True
                        #Weakest link in the SVM fundamentally
                        for i in self.data:
                            for xi in self.data[i];
                            yi = 1
                            if not yi*(np.dot(w_t, xi) + b) >= 1:
                                found_opcion = False
                        if found_opcion:
                            opt_dict = [np.linalg.norm(w_t)] = [w_,t,b]
                if( w[0]  > 0):
                    optimized = True
                    print"Optimized a step."
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict]) #From lowest to highest

            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        # sign (w.w+b)
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        return classification

data_dict = { -1:np.array([[1,7],
                           [2,8],
                           [3,8],]),
                1:np.array([[5,1],
                            [6,-1],
                            [7,3],])}
