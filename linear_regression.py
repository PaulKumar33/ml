import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


class linear_regression:
    def __init__(self):
        self.global_theta = []
        pass

    def linear_estimator(self, x_in, theta):
        '''h_theta calculations. theta element of 1xd, x an element of 1xd, yn is 1x1'''
        '''this works on a single point'''
        sum = 0
        sum += np.dot(theta, np.transpose(x_in))
        return sum

    def LMS(self, x_in, y_in, theta):
        '''return the lms value for a single number y_in, a 1xd vector x_in, 1xd vector theta, and learning rate alpha'''
        err = (y_in - self.linear_estimator(x_in, theta))
        #print(err)
        err = err*x_in
        #note err should be a 1xd vector

        return err

    def gradient_descent_batch(self, x_vect, theta_vect, y_vector, alpha, iterations):
        '''x vector is the features, theta are the weights'''
        '''let the element 0 of the theta vector to be the intercept'''
        current_iter = 0
        current_theta = theta_vect

        while(current_iter <iterations):
            error_sum = 0
            for i in range(len(x_vect)):
                err = self.LMS(x_vect[i], y_vector[i], current_theta)
                error_sum += alpha*err
            current_theta = current_theta + error_sum
            current_iter += 1
        self.global_theta = current_theta


if __name__ == "__main__":
    linear_reg = linear_regression()
    '''x_test = np.array([[1,3]])
    theta_test = np.array([1,1])
    y_test = np.array([[2]])'''
    x = [0.01,
0.0397,
0.066527,
0.08056049,
0.118814462,
0.123525534,
0.14399449,
0.154325453,
0.157870664,
0.180907617,
0.182869825,
0.198544452,
0.200311686,
0.19841101,
0.213549404,
0.214081491,
0.227265196,
0.224586888,
0.219858174,
0.230897491]
    y=[0.01,
0.0694,
0.176708,
0.21880847,
0.410078328,
0.4147894,
0.455727313,
0.497051164,
0.507686795,
0.622871563,
0.624833772,
0.656183024,
0.663251962,
0.657549935,
0.733241901,
0.733773988,
0.760141398,
0.749428167,
0.735242025,
0.79043861,
]
    alpha = 0.02
    theta = np.array([1,1])

    x_arr = []
    y_arr = []
    for i in range(len(x)):
        x_arr.append([1, x[i]])
        y_arr.append([y[i]])
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    #print(x_arr[0][1])
    linear_reg.gradient_descent_batch(x_arr, theta, y_arr, alpha, 4000)
    theta_global = linear_reg.global_theta
    print(x_arr)

    x_plot = []
    y_plot = []
    for j in range(len(x_arr)):
        x_plot.append(x_arr[j][1])
        y_plot.append(np.dot(x_arr[j], np.transpose([theta_global])))
    print(y_plot)
    plt.plot(x_plot, y_plot)
    plt.plot(x_plot, y)
    plt.show()

    #linear_reg.LMS(x_test[0], y_test[0], theta_test, alpha)