import pandas as pd


# from matplotlib import pyplot


def read_data():
    """This function reads the contents of the csv file from the file directory and file name given as input by the
       user and returns a pandas dataframe object"""

    file_path = str(input("Enter the file path where the data is located: "))
    file_name = str(input("Enter the csv file name which is to be read: "))
    csvfile = file_path + "/" + file_name + ".csv"
    csvdata = pd.read_csv(csvfile)
    del csvdata[csvdata.columns[0]]
    return csvdata


def format_data(data, reqsize, K):
    """This function takes the pandas dataframe, required training dataset size and no. of folds for K fold
       cross - validation and splits the dataframe into two outputs: training data which is to be fed into the
       Kfold_cv function and testing data"""


    shuffleddata = data.sample(frac=1, random_state=42)
    totsize = reqsize + (reqsize // (K - 1))
    test_n = int(0.25 * totsize)
    unfil_train = shuffleddata.sample(n=totsize, random_state=42)
    nocv_data = unfil_train.copy()
    nocv_data = nocv_data.reset_index(drop=True)
    exceptsampledata = shuffleddata.drop(unfil_train.index)
    test = exceptsampledata.sample(n=test_n, random_state=42)
    retest = test.copy()
    retest = retest.reset_index(drop=True)
    print("The length of the split training data before sending to K fold splitting : ", len(nocv_data))
    print("The length of the split testing data : ", len(retest))

    return nocv_data, retest


def kfold_cv(K, data):
    """ Performs K-fold data splits in the data given and returns the training and validation fold splits"""

    fold_length = data.shape[0] // K
    val_list = []
    train_list = []
    end = data.shape[0]

    for i in range(K):
        val = data[i * fold_length:(i + 1) * fold_length]
        if i == 0:
            train1 = data[0:0]
        else:
            train1 = data[0:(i * fold_length)]
        train2 = data[(i + 1) * fold_length:end]
        train = pd.concat([train1, train2], axis=0)
        train_list.append(train)
        val_list.append(val)

    return train_list, val_list
