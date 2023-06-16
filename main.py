import pandas as pd
from DataForm import read_data, format_data, kfold_cv
from matplotlib import pyplot
from Regression import Regression


"""Intializing all the variables"""

# input_path = "C:\Users\laxman\OneDrive\Desktop\ML\PRMLv2_Task1"
# input_filename = "function2"
train_datasize = 50
K = 4
degrees = [2, 3, 6]
degree_data = {}
ERMS_totallist = []
lambdas = [0, 1.5e-5, 0.51e-2, 1, 4, 8]
MERMS_train = 0
MERMS_val = 0
fold = (K // 2) - 1
grid_size = 100

""" read_data() reads the csv file from the directory and filename that the user inputs and returns the csv file data 
as it is """

unshuffleddata = read_data()

""" format_data() shuffles the data and gives out training data of size = (train_datasize//(K-1)) + train_datasize 
and testing data of size = 25 % of datasize """

nocvfold_data, test_data = format_data(unshuffleddata, train_datasize, K)

""" Returns required splits of the K fold and after the split we see the size of the training matches that of 
train_datasize variable. This is done for even splitting in cross - validation """

train_list, val_list = kfold_cv(K, nocvfold_data)

for degree in degrees:

    lambda_data = {}

    for lda in lambdas:

        SERMS_train = 0
        SERMS_val = 0
        SERMS_test = 0
        fold_data = {}

        for i in range(K):

            train_data = train_list[i].copy()
            val_data = val_list[i].copy()

            Trial = Regression(train_data, degree, True, lda)
            W, X_train1, Y_train1 = Trial.train_w()

            ERMS_train, X_train, Y_train, Y_trainpred = Trial.test_w(W, train_data)
            ERMS_val, X_val, Y_val, Y_valpred = Trial.test_w(W, val_data)
            ERMS_test, X_test, Y_test, Y_testpred = Trial.test_w(W, test_data)

            # Forming the mesh for plotting the predicted curve or surface

            X_plot_train, Y_plot_train = Trial.mesh_formation(X_train, degree, grid_size, W)
            X_plot_val, Y_plot_val = Trial.mesh_formation(X_val, degree, grid_size, W)
            X_plot_test, Y_plot_test = Trial.mesh_formation(X_test, degree, grid_size, W)

            SERMS_train += ERMS_train
            SERMS_val += ERMS_val
            SERMS_test += ERMS_test

            fold_data[i + 1] = [W, ERMS_train, ERMS_val, ERMS_test, X_train, Y_train, Y_trainpred,
                                X_plot_train, Y_plot_train, X_val, Y_val, Y_valpred, X_plot_val,
                                Y_plot_val, X_test, Y_test, Y_testpred, X_plot_test,
                                Y_plot_test]

            if lda == 0 and i == fold:

                X_train0 = fold_data[i + 1][4]
                Y_train0 = fold_data[i + 1][5]
                X_plot_train0 = fold_data[i + 1][7]
                Y_plot_train0 = fold_data[i + 1][8]

            if i == fold:
                main_title = "Polynomial Curve Fitting on Training data: Degree = %d; Fold = %d; ERMS = %f;" % (
                    degree, i + 1, ERMS_train)

                Trial.plottingforregression(main_title, lda, X_train, Y_train, X_plot_train,
                                            Y_plot_train, X_train0, Y_train0, X_plot_train0, Y_plot_train0)

                # error_title = 'Error Analysis for degree = %d and Lambda = %f' % (degree, lda)
                # Trial.error_analysis_plot()

        MERMS_train = SERMS_train / K
        MERMS_val = SERMS_val / K
        MERMS_test = SERMS_test / K

        ERMS_data = [degree, lda, MERMS_train, MERMS_val, MERMS_test]
        ERMS_totallist.append(ERMS_data)

        lambda_data[lda] = fold_data

    degree_data[degree] = lambda_data

ERMS_table = pd.DataFrame(ERMS_totallist, columns=['Degree', 'Lambda', 'Mean ERMS of Training data',
                                                   'Mean ERMS of Validation data', 'Mean ERMS of Testing data'])

Best_model_predict = ERMS_table.loc[ERMS_table['Mean ERMS of Validation data'].idxmin()]
Best_model_true = ERMS_table.loc[ERMS_table['Mean ERMS of Testing data'].idxmin()]

csvtitle = 'Regression1D_Error_table_training_size_%d.csv' % train_datasize
ERMS_table.to_csv(csvtitle, encoding='utf-8')
print(Best_model_predict)
print(Best_model_true)

pyplot.show()