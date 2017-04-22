import sys
sys.path.append("..")
import utils
from utils import *
from softmax_skeleton import softmaxRegression, getClassification, plotCostFunctionOverTime, computeTestError, computeTestErrorMod3, updateY
import numpy as np
import matplotlib.pyplot as plt
from features import *
import time

## Load MNIST data:
#trainX, trainY, testX, testY = getMNISTData()
# Plot the first 20 images of the training set.
#plotImages(trainX[0:20,:])  

# runSoftmaxOnMNIST: trains softmax, classifies test data, computes test error, and plots cost function
def runSoftmaxOnMNIST():
    print("[INFO] Running soft max on MNIST")
    trainX, trainY, testX, testY = getMNISTData()
    print("[INFO] Loaded in data.")
    print("[INFO] Using tempParameter of:", tempParameter)

    start = time.time()
    theta, costFunctionHistory = softmaxRegression(trainX, trainY, tempParameter, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
    finish = time.time()

    print("[INFO] Finished training in %f secs" % (finish-start))

    #plotCostFunctionOverTime(costFunctionHistory)
    testError = computeTestError(testX, testY, theta, tempParameter)
    print("Test Error:", testError)

    # Save the model parameters theta obtained from calling softmaxRegression to disk.
    writePickleData(theta, "./theta.pkl.gz")  
    
    # compute error mod 3
    trainYMod3, testYMod3 = updateY(trainY, testY)
    testErrorMod3 = computeTestErrorMod3(testX, testYMod3, theta, tempParameter)
    print("Test Error (mod 3):", testErrorMod3)

    return (testError, testErrorMod3)

# Part 1: Step 6
# Try 3 different values of temperature parameter
tempParameter = 1.0
def tryDifferentTempParams():
    for tau in [0.5, 1.0, 2.0]:
        global tempParameter
        tempParameter = tau
        print('testError =', runSoftmaxOnMNIST())

#tempParameter = 1
#runSoftmaxOnMNIST()
#print('testError =', runSoftmaxOnMNIST()) 


# runSoftmaxOnMNISTMod3: trains Softmax regression on digit (mod 3) classifications
def runSoftmaxOnMNISTMod3():
    """
    -Runs softmaxRegression on MNIST training set with new trainYMod3
    -computes test error by running computeTestErrorMod3()
    -plots cost function over number of iters
    -save new model to file: thetaM.pkl.gz
    """
    print("[INFO] Running soft max on MNISTMod3")
    trainX, trainY, testX, testY = getMNISTData()
    trainYMod3, testYMod3 = updateY(trainY, testY)

    print("[INFO] Loaded in data (mod3).")
    print("[INFO] Using tempParameter of:", tempParameter)

    start = time.time()
    theta, costFunctionHistory = softmaxRegression(trainX, trainYMod3, tempParameter, alpha=0.3, lambdaFactor=1.0e-4, k=10, numIterations=150)
    finish = time.time()

    print("[INFO] Finished training in %f secs" % (finish-start))

    plotCostFunctionOverTime(costFunctionHistory)

    testErrorMod3 = computeTestErrorMod3(testX, testYMod3, theta, tempParameter)

    # Save the model parameters theta obtained from calling softmaxRegression to disk.
    writePickleData(theta, "./thetaM.pkl.gz")  
         
    return testErrorMod3


# TODO: Run runSoftmaxOnMNISTMod3(), report the error rate
#testErrorMod3 = runSoftmaxOnMNISTMod3()
#print("Test error mod 3:", testErrorMod3)
                                
                                
######################################################
# This section contains the primary code to run when
# working on the "Using manually crafted features" part of the project.
# You should only work on this section once you have completed the first part of the project.
######################################################
## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.

# Load MNIST data:
def runSoftmaxOnMNIST_PCA():
    """
    print("[INFO] Running soft max on MNIST")
    trainX, trainY, testX, testY = getMNISTData()
    print("[INFO] Loaded in data.")
    print("[INFO] Using tempParameter of:", tempParameter)

    start = time.time()
    theta, costFunctionHistory = softmaxRegression(trainX, trainY, tempParameter, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
    finish = time.time()

    print("[INFO] Finished training in %f secs" % (finish-start))

    #plotCostFunctionOverTime(costFunctionHistory)
    testError = computeTestError(testX, testY, theta, tempParameter)
    print("Test Error:", testError)
    """  
    tempParameter = 1.0
    print("[INFO] Running soft max on MNIST data with PCA")
    print("[INFO] Using tempParameter of:", tempParameter)

    trainX, trainY, testX, testY = getMNISTData()
    print("[INFO] Loaded in data...")

    n_components = 18
    pcs = principalComponents(trainX)
    train_pca = projectOntoPC(trainX, pcs, n_components)
    test_pca = projectOntoPC(testX, pcs, n_components)
    print(test_pca)
    print("[INFO] Projected data into PCA feature dimension...")

    # # TODO: Train your softmax regression model using (train_pca, trainY) 
    # #       and evaluate its accuracy on (test_pca, testY).
    start = time.time()

    print("[INFO] Dims of train_pca and trainY:",np.shape(train_pca), np.shape(trainY))
    print("[INFO] Dims of test_pca and testY:",np.shape(test_pca), np.shape(testY))
    theta, costFunctionHistory = softmaxRegression(train_pca, trainY, tempParameter, alpha=0.3, lambdaFactor=1.0e-4, k=10, numIterations=40)
    finish = time.time() # 0.9244

    print("[INFO] Finished training in %f secs" % (finish-start))

    # plot and compute error
    plotCostFunctionOverTime(costFunctionHistory)

    print("Temp Param:", tempParameter)
    testError = computeTestError(test_pca, testY, theta, tempParameter)
    print("Test Error (PCA):", testError)

    #TODO: Use the plotPC function in features.py to produce scatterplot
          #of the first 100 MNIST images, as represented in the space spanned by the 
          #first 2 principal components found above.
    plotPC(trainX[range(100),], pcs, trainY[range(100)])

    #TODO: Use the reconstructPC function in features.py to show
          #the first and second MNIST images as reconstructed solely from 
          #their 18-dimensional principal component representation.
          #Compare the reconstructed images with the originals.
    firstimage_reconstructed = reconstructPC(train_pca[0,], pcs, n_components, trainX)
    plotImages(firstimage_reconstructed)
    plotImages(trainX[0,])

    secondimage_reconstructed = reconstructPC(train_pca[1,], pcs, n_components, trainX)
    plotImages(secondimage_reconstructed)
    plotImages(trainX[1,]) 

#runSoftmaxOnMNIST_PCA()


def runSoftmaxOnMNIST_PCA_Cubic():
    """
    print("[INFO] Running soft max on MNIST")
    trainX, trainY, testX, testY = getMNISTData()
    print("[INFO] Loaded in data.")
    print("[INFO] Using tempParameter of:", tempParameter)

    start = time.time()
    theta, costFunctionHistory = softmaxRegression(trainX, trainY, tempParameter, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
    finish = time.time()

    print("[INFO] Finished training in %f secs" % (finish-start))

    #plotCostFunctionOverTime(costFunctionHistory)
    testError = computeTestError(testX, testY, theta, tempParameter)
    print("Test Error:", testError)
    """
    print("[INFO] Running soft max on MNIST data with PCA 10 and Cubic Kernel...")

    tempParameter = 1.0
    print("[INFO] Using tempParameter of:", tempParameter)

    ### LOAD DATA ###
    trainX, trainY, testX, testY = getMNISTData()
    print("[INFO] Loaded in data...")

    ### Find the 10-dimensional PCA representation of the training and test set ###
    n_components = 10
    pcs = principalComponents(trainX)
    train_pca10 = projectOntoPC(trainX, pcs, n_components)
    test_pca10 = projectOntoPC(testX, pcs, n_components)
    print("[INFO] Got PCA features...")

    ### CONVERT TO CUBIC REPRESENTATION ###
    train_cube = cubicFeatures(train_pca10)
    test_cube = cubicFeatures(test_pca10)
    print("[INFO] Got cubic features...")

    ### TRAIN THE MODEL ###
    start = time.time()
    theta, costFunctionHistory = softmaxRegression(train_cube, trainY, tempParameter, alpha=0.3, lambdaFactor=1.0e-4, k=10, numIterations=150)
    finish = time.time()
    print("[INFO] Finished training in %f secs" % (finish-start))

    ### PLOTTING ###
    plotCostFunctionOverTime(costFunctionHistory)

    ### COMPUTE ERROR ###
    err = computeTestError(test_cube, testY, theta, tempParameter)
    print(" ### TEST ERROR RESULTS ###")
    print("Test Error (PCA10 w/ Cubic Features):", err)

    return err

#err = runSoftmaxOnMNIST_PCA_Cubic()