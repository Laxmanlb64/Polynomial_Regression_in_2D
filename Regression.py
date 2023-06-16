import numpy as np
from DesignMatrixfunction import designmatrix_gen
from matplotlib import pyplot
import math

class Regression:
    def __init__(self, train_data, degree, regularisation, lda):
        self.train_data = train_data
        self.degree = degree
        self.lda = lda
        self.regularisation = regularisation

    def train_w(self):

        degree = self.degree
        all_train = self.train_data.copy()
        all_train.drop(columns=all_train.columns[-1], axis=1, inplace=True)
        train_x = all_train.copy()
        train_y = self.train_data.iloc[:, -1]

        train_x_np = train_x.to_numpy()
        train_y_np = train_y.to_numpy()

        num = train_x_np.shape[0]
        feature_size = train_x_np.shape[1]
        monomials = math.factorial(degree + feature_size) // (math.factorial(degree) * math.factorial(feature_size))


        # Design matrix formation

        # print(train_x_np.shape)

        DM = designmatrix_gen(train_x_np, degree)

        print("Training DM formed")

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

        feature_size = xtrain.shape[1]

        if feature_size == 1:

            pyplot.figure()
            pyplot.title(main_title)
            pyplot.scatter(xtrain, ytrain, c='snow', edgecolors='blue', label="Ground Truth (Training)")

            if lda != 0:
                pyplot.plot(xplottrain, yplottrain
                            , 'green', label="Predicted curve with regularisation (lambda = %f)" % (lda))

            pyplot.plot(xplottrain0, yplottrain0, 'red', label="Predicted curve without regularisation")
            pyplot.legend(loc="upper left")
            pyplot.xlabel('x')
            pyplot.ylabel('y')

        elif feature_size == 2:

            fig = pyplot.figure(figsize=(14, 9))
            ax = pyplot.axes(projection='3d')

            # Creating plot
            ax.plot_surface(xplottrain[0], xplottrain[1], yplottrain, cmap = 'viridis' )
            ax.plot(xtrain[:, 0], xtrain[:, 1], ytrain, 'ro', label='Ground Truth', zorder=4, markersize=5)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')


        else:
            print("Can't form Design matrix for input data size > 2 and input data size is %d" % feature_size)


    # def error_analysis_plot(self, train_erms_list, val_erms_list, degrees, title):
    #
    #     pyplot.plot(degrees, train_erms_list, label = "Training RMS error")
    #     pyplot.plot(degrees, val_erms_list, label = 'Validation ERMS error')
    #     pyplot.xlabel("Model complexity in degrees")
    #     pyplot.ylabel("Error")
    #     # pyplot.show()
