import numpy as np
import math


def designmatrix_gen(x, degree):

    num = x.shape[0]
    feature_size = x.shape[1]


    if feature_size == 1:
        '''This if condition checks whether the input is 1D if yes, it will proceed to form the design matrix using 
           1D inputs, applicable to 1D regression'''

        DesignMatrix = np.ones([num, degree + 1])

        for j in range(1, degree + 1):
            DesignMatrix[:, j] = DesignMatrix[:, j] * (x[:, 0] ** j)

        return DesignMatrix

    elif feature_size == 2:
        '''This if condition checks whether the input is 2D if yes, it will proceed to form the design matrix using 
           1D inputs, applicable to 2D regression'''

        x1 = x[:, 0].copy()
        x2 = x[:, 1].copy()
        monomials = math.factorial(degree + feature_size) // (math.factorial(degree) * math.factorial(feature_size))


        DesignMatrix = np.ones([num, monomials])

        count = 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                product = np.power(x1, i) * np.power(x2, j)
                DesignMatrix[:, count] = product
                count += 1


        return DesignMatrix


    else:
        print("Can't form Design matrix for input data size > 2 and input data size is %d" % feature_size)




def linear_model(x, degree, w):

    DesignMatrix = designmatrix_gen(x, degree)
    y = np.matmul(DesignMatrix, w)

    return y
