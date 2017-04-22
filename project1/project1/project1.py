"""
Written by Milo Knowles.
March 3, 2017.
"""

from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt
import random

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    numDataPoints = np.shape(feature_matrix)[0] # num rows in feature_matrix

    totalLoss = 0
    for i in range(numDataPoints): # for each data point
        # working with data point i
        error = labels[i] * (np.dot(theta, feature_matrix[i])+theta_0)
        totalLoss += (1-error) if error <= 1 else 0

    return float(totalLoss) / numDataPoints



def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    theta = current_theta + label*feature_vector
    theta_0 = current_theta_0 + label
    return (theta, theta_0)
    

def perceptron(feature_matrix, labels, T):
    """
    Section 1.4a
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    numDataPoints = np.shape(feature_matrix)[0] # num rows in feature_matrix
    theta = np.zeros(np.shape(feature_matrix)[1]) # intially filled with zeros
    theta0 = 0

    for t in range(T):
        for i in range(numDataPoints):

            # if point is misclassified
            if (labels[i]*(np.dot(theta, feature_matrix[i])+theta0)) <= 0:
                # call our helper function to update theta and theta0
                theta, theta0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta0)
                
            # otherwise, continue
            else:
                continue

    return (theta, theta0)
            
    
def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.4b
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    thetaSum = np.zeros(np.shape(feature_matrix)[1])
    thetaNaughtSum = 0

    numDataPoints = np.shape(feature_matrix)[0] # num rows in feature_matrix
    theta = np.zeros(np.shape(feature_matrix)[1]) # intially filled with zeros
    theta0 = 0

    numUpdates = 0
    for t in range(T):
        for i in range(numDataPoints):
            # if point is misclassified
            if (labels[i]*(np.dot(theta, feature_matrix[i])+theta0)) <= 0:

                numUpdates += 1

                # call our helper function to update theta and theta0
                theta, theta0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta0)

                # add to our thetaSum total
                thetaSum += theta
                thetaNaughtSum += theta0
                
            else:
                numUpdates += 1
                thetaSum += theta
                thetaNaughtSum += theta0

    avgTheta = thetaSum / numUpdates
    avgTheta0 = float(thetaNaughtSum) / numUpdates
    return (avgTheta, avgTheta0)
    

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    error = label * (np.dot(current_theta, feature_vector)+current_theta_0)

    if error <= 1:
        newTheta = (1-L*eta)*current_theta + eta*label*feature_vector
        newTheta0 = (1-L*eta)*current_theta_0 + eta*label

    else:
        newTheta = (1-L*eta)*current_theta
        newTheta0 = (1-L*eta)*current_theta_0

    return (newTheta, newTheta0)


def pegasos(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    
    For each update, set learning rate = 1/sqrt(t), 
    where t is a counter for the number of updates performed so far (between 1 
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    numDataPoints = np.shape(feature_matrix)[0] # num rows in feature_matrix
    theta = np.zeros(np.shape(feature_matrix)[1]) # intially filled with zeros
    theta0 = 0
    possiblePoints = range(numDataPoints)

    counter = 0
    for t in range(T):
        for n in range(numDataPoints):
            counter += 1

            # select i at random
            i = random.choice(possiblePoints)

            # update eta based on number of timesteps
            eta = 1 / (counter**0.5)

            # do update step
            theta, theta0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta0)

    return (theta, theta0)

### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    numDataPoints = np.shape(feature_matrix)[0]
    predictions = np.zeros(numDataPoints)

    for i in range(numDataPoints):
        result = np.dot(theta, feature_matrix[i]) + theta_0
        predictions[i] = -1 if result <= 0 else 1

    return predictions

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    # train the classifier
    theta_perceptron, theta0_perceptron = perceptron(train_feature_matrix, train_labels, T)

    # get classifications from our classifier
    train_classif = classify(train_feature_matrix, theta_perceptron, theta0_perceptron)
    valid_classif = classify(val_feature_matrix, theta_perceptron, theta0_perceptron)

    # determine accuracy
    train_accuracy = accuracy(train_classif, train_labels)
    valid_accuracy = accuracy(valid_classif, val_labels)

    return (train_accuracy, valid_accuracy)


def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    # train the classifier
    theta_avg_perceptron, theta0_avg_perceptron = average_perceptron(train_feature_matrix, train_labels, T)

    # get classifications from our classifier
    train_classif = classify(train_feature_matrix, theta_avg_perceptron, theta0_avg_perceptron)
    valid_classif = classify(val_feature_matrix, theta_avg_perceptron, theta0_avg_perceptron)

    # determine accuracy
    train_accuracy = accuracy(train_classif, train_labels)
    valid_accuracy = accuracy(valid_classif, val_labels)

    return (train_accuracy, valid_accuracy)


def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    # train the classifier
    theta_pegasos, theta0_pegasos = pegasos(train_feature_matrix, train_labels, T, L)

    # get classifications from our classifier
    train_classif = classify(train_feature_matrix, theta_pegasos, theta0_pegasos)
    valid_classif = classify(val_feature_matrix, theta_pegasos, theta0_pegasos)

    # determine accuracy
    train_accuracy = accuracy(train_classif, train_labels)
    valid_accuracy = accuracy(valid_classif, val_labels)

    return (train_accuracy, valid_accuracy)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def extract_words_no_punctuation(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

        # allow ! and ? but no other symbols
        #'0', '1', '2', '3', '4', '5', '6' ,'7' ,'8', '9'
        for symbol in ["'", ",", ".", '_', '-', '=', '+', '<', '>', "&", "*", "%", "$", "@", "#", '/', ':', '"']:
            input_string = input_string.replace(symbol, '')

    #print(input_string)
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def bag_of_words_removed_stopwords(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    DOES NOT ADD STOPWORDS TO THE DICTIONARY

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index

    stopwords = {}
    with open('stopwords.txt', 'r') as f:
        for line in f:
            line = line[0:-1] # get rid of the \n at the end of each line
            #line.replace('\n', '')
            if line not in stopwords:
                stopwords[line] = True

    for text in texts:
        word_list = extract_words(text)
        #print(word_list)
        #assert False
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary


def bag_of_words_removed_stopwords_and_punctuation(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    DOES NOT ADD STOPWORDS TO THE DICTIONARY

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index

    stopwords = {}
    with open('stopwords.txt', 'r') as f:
        for line in f:
            line = line[0:-1] # get rid of the \n at the end of each line
            #line.replace('\n', '')
            if line not in stopwords:
                stopwords[line] = True

    for text in texts:
        word_list = extract_words_no_punctuation(text)
        #print(word_list)
        #assert False
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        #print(word_list)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def extract_bow_feature_vectors_with_frequency(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Normalizes all feature vectors to have unit length!
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])


    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1.0

    # row_norm = np.linalg.norm(feature_matrix, axis=1, ord=4)
    # feature_matrix_normalized = feature_matrix / row_norm.reshape(num_reviews,1)
    #feature_matrix_normalized = feature_matrix

    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    num_extra_features = 1 # change this as more features added
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, num_extra_features])

    max_review_length = 0
    # make the length of the review a feature
    for i, text in enumerate(reviews):
        word_list = extract_words_no_punctuation(text)
        if len(word_list) > max_review_length:
            max_review_length = len(word_list)
        feature_matrix[i, 0] = float(len(word_list))

    for i in range(num_reviews):
        feature_matrix[i, 0] /= max_review_length

    return feature_matrix

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors_with_frequency(reviews, dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    # result = (preds == targets)
    # #print(type(result))
    # print(np.shape(preds), len(targets))
    return (preds == targets).mean()
