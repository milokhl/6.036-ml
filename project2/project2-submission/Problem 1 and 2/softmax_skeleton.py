import sys
sys.path.append("..")
import utils
from utils import *
import math
import numpy as np
import matplotlib.pyplot as plt
import time

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    """
    X: n x d (n data points, d features)
    theta: k x d (row j represents params for label j)
    tempParameter: a scalar

    Returns: H, a k x n matrix
    """
    s = time.time()
    H = np.divide(np.dot(theta, np.transpose(X)), float(tempParameter))

    # find max item in each column, create c vector, and subtract off
    c = np.max(H, 0)
    for j in range(np.shape(H)[0]):
        H[j] = H[j] - c

    # exponentiate the matrix
    H = np.exp(H)

    computeProbabilitiesResult = np.divide(H, H.sum(axis=0, keepdims=True))

    #print("computeProbabilities: %f" % (time.time()-s))
    return computeProbabilitiesResult

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    s = time.time()

    computeProbabilitiesResult = computeProbabilities(X, theta, tempParameter)

    # take the ln of every element
    H = np.log(computeProbabilitiesResult)

    # apply the sign function
    for i in range(np.shape(X)[0]): # from 0 to n-1
        for j in range(np.shape(theta)[0]): # from 0 to k-1
            if not Y[i]==j:
                H[j][i] = 0

    # sum and normalize by n
    H = np.divide(np.sum(H), -np.shape(X)[0])

    theta_squared_sum = np.sum(np.square(theta))
    J_theta = H + lambdaFactor * theta_squared_sum / 2.0

    #print("computeCostFunction: %f" % (time.time()-s))
    return J_theta


def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
    s = time.time()

    computeProbabilitiesResult = computeProbabilities(X, theta, tempParameter)

    gradTheta = np.zeros(np.shape(theta)) # has same dims as theta
    for j in range(np.shape(theta)[0]): # for each row in theta

        # compute the partial derivative wrt theta[j]
        vectSum = np.zeros(np.shape(X)[1])
        for i in range(np.shape(X)[0]): # for each example
            p = computeProbabilitiesResult[j][i]
            if Y[i]==j:
                vectSum = np.add(vectSum, np.multiply(X[i], (1.0-p)))
            else:
                vectSum = np.add(vectSum, np.multiply(-p, X[i]))
        # set the row in gradTheta to the vector we just calculated
        gradTheta[j] = np.add(np.multiply((-1.0 / (tempParameter*np.shape(X)[0])), vectSum), np.multiply(lambdaFactor, theta[j]))

    # update theta
    thetaNew = np.subtract(theta, np.multiply(alpha, gradTheta))
    #print("runGradDesc: %f" % (time.time()-s))
    return thetaNew

def updateY(trainY, testY):
    """
    Returns: trainYMod3 (nx1), testYMod3 (nx1)
    """
    return (trainY % 3, testY % 3)

def computeTestErrorMod3(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    assignedLabels = assignedLabels % 3 # mod by 3 before comparison
    return 1 - np.mean(assignedLabels == Y)

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    startTime = time.time()

    for i in range(numIterations):
        print("Iteration %d started at %f secs" % (i, time.time() - startTime))

        time1 = time.time()
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))

        time2 = time.time()
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
        #print(theta)

    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
