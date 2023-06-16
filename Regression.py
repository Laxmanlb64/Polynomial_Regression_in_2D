import numpy as np
from DesignMatrixfunction import designmatrix_gen, linear_model
import matplotlib as mpl
from matplotlib import pyplot
import math


class Regression:
    def __init__(self, train_data, degree, regularisation, lda):
        self.train_data = train_data
        self.degree = degree
        self.lda = lda
        self.regularisation = regularisation

    def train_w(self):
        """ This function trains the model based on the input data given when intializing the class object. This
             function returns the optimal weight parameter, input data for training and
             ground truth output data (This is not predicted output data)"""

        degree = self.degree
        all_train = self.train_data.copy()
        all_train.drop(columns=all_train.columns[-1], axis=1, inplace=True)
        train_x = all_train.copy()
        train_y = self.train_data.iloc[:, -1]

        train_x_np = train_x.to_numpy()
        train_y_np = train_y.to_numpy()

        feature_size = train_x_np.shape[1]
        monomials = math.factorial(degree + feature_size) // (math.factorial(degree) * math.factorial(feature_size))

        # Design matrix formation

        DM = designmatrix_gen(train_x_np, degree)

        if self.regularisation is True:

            lda = self.lda

        else:

            lda = 0

        DM_T = np.transpose(DM)
        DM_TxDM = np.matmul(DM_T, DM)
        lambdamat = lda * np.identity(monomials)
        pseudoreg = DM_TxDM + lambdamat
        pseudoinv = np.linalg.inv(pseudoreg)
        DM_TxY = np.matmul(DM_T, train_y_np)
        opt_w = np.matmul(pseudoinv, DM_TxY)

        return opt_w, train_x_np, train_y_np

    def test_w(self, w, test_data):
        """ This functions returns the predicted output based on the weight parameter given and input test data
            But only the inputs of the test data will be used to predict the predicted output. This function
            also retunrs the RMSE of the predicted output, ground truth output and input"""

        degree = self.degree
        all_test = test_data.copy()
        all_test.drop(columns=all_test.columns[-1], axis=1, inplace=True)
        test_x = all_test.copy()
        test_y = test_data.iloc[:, -1]

        test_x_np = test_x.to_numpy()
        test_y_np = test_y.to_numpy()

        # Design matrix formation
        DM_test = designmatrix_gen(test_x_np, degree)

        y_pred = np.matmul(DM_test, w)

        SE = (test_y_np - y_pred) ** 2

        SE = SE.reshape((SE.shape[0], 1))
        MSE = np.mean(SE)
        RMSE = np.sqrt(MSE)

        return RMSE, test_x_np, test_y_np, y_pred

    def plottingforregression(self, main_title, lda, xtrain, ytrain, xplottrain, yplottrain,
                              xtrain0=None, ytrain0=None, xplottrain0=None, yplottrain0=None):
        """This function plots the predicted curve or surface based on the inputs given"""

        feature_size = xtrain.shape[1]

        if feature_size == 1:

            pyplot.figure()
            pyplot.title(main_title)
            pyplot.scatter(xtrain, ytrain, c='snow', edgecolors='blue', label="Ground Truth (Training)")

            if lda != 0:
                pyplot.plot(xplottrain, yplottrain
                            , 'green', label="Predicted curve with regularisation (lambda = %f)" % lda)

            pyplot.plot(xplottrain0, yplottrain0, 'red', label="Predicted curve without regularisation")
            pyplot.legend(loc="upper left")
            pyplot.xlabel('x')
            pyplot.ylabel('y')

        elif feature_size == 2:

            mpl.rcParams['legend.fontsize'] = 10
            fig = pyplot.figure(figsize=(14, 9))
            ax = pyplot.axes(projection='3d')

            # Creating plot
            ax.plot_surface(xplottrain[0], xplottrain[1], yplottrain, cmap='viridis')
            ax.plot(xtrain[:, 0], xtrain[:, 1], ytrain, 'ro', label='Ground Truth', zorder=4, markersize=5)
            ax.text2D(0.05, 0.95, "Regularisation parameter (lambda) = %f" % lda, transform=ax.transAxes)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')
            ax.set_title(main_title)
            ax.legend()

        else:
            print("Can't form Design matrix for input data size > 2 and input data size is %d" % feature_size)

    # def error_analysis_plot(self, train_erms_list, val_erms_list, degrees, title):
    #
    #     pyplot.plot(degrees, train_erms_list, label = "Training RMS error")
    #     pyplot.plot(degrees, val_erms_list, label = 'Validation ERMS error')
    #     pyplot.xlabel("Model complexity in degrees")
    #     pyplot.ylabel("Error")
    #     # pyplot.show()

    def mesh_formation(self, X, degree, grid_size, W):
        """This function forms the Mesh grid which is to be used for plotting the fitted curve or surface"""

        feature_size = X.shape[1]

        if feature_size == 1:

            X_plot = np.linspace(np.min(X), np.max(X), grid_size)
            X_plot = X_plot.reshape([grid_size, 1])
            Y_plot = linear_model(X_plot, degree, W)

            return X_plot, Y_plot

        elif feature_size == 2:

            Y_plot = []

            # Here X is 2D input vector or numpy array X = [(x10,x20),(x11,x21),...,(x1n,x2n)]
            '''We are finding the minimum and maximum of all the possible elements in the 2D numpy array X'''

            X1 = np.linspace(np.min(X), np.max(X), grid_size)
            X2 = X1.copy()

            # Forming the x1, x2 grid mesh for finding the corresponding output values for surface plot
            ''' X is returned as an 3D numpy array with dimensions (2, grid_size, grid_size)
                [
                   [ [x10, x10, x10,......., x10],
                     [x11, x11, x11,......., x11],  
                     [x12, x12, x12,......., x12],
                      ....
                      ....
                     [x1n, x1n, x1n,......., x1n]  ],
                     
                   [ [x10, x10, x10,......., x10],
                     [x11, x11, x11,......., x11],  
                     [x12, x12, x12,......., x12],
                      ....
                      ....
                     [x1n, x1n, x1n,......., x1n]  ]  
                ]   '''

            X1_plot, X2_plot = np.meshgrid(X1, X2)
            X_plot = np.array([X1_plot.T, X2_plot.T])

            # Forming the output values for the formed x1, x2 mesh grid
            ''' For each x1 value, we are looping through all the x2 values and finding the function output array
                (Y_plot_train) as:

                [[f(x10,x20), f(x10,x21), f(x10, x22).....f(x10,x2n)]
                [f(x11,x20), f(x11,x21), f(x11, x22).....f(x11,x2n]
                ....
                ....
                [f(x1n,x20), f(x1n,x21), f(x1n, x22).....f(x1n,x2n]]

                 x1l corresponds to the lth value in the x1 input array
                 x2m corresponds to the mth value in the x2 input array'''

            for l in range(grid_size):

                X_column = []

                for m in range(grid_size):
                    X_column.append(np.array([X1[l], X2[m]]))
                X_column = np.array(X_column)
                Y_column = linear_model(X_column, degree, W)
                Y_plot.append(Y_column)

            Y_plot = np.array(Y_plot)

            return X_plot, Y_plot

        else:

            print("Feature size is too large. So Mesh grid cannot be expressed in the 3D cartesian coordinate system")
