import numpy as np
import matplotlib.pyplot as plt

# For all these functions, X = n x d numpy array containing the training data
# where each row of X = one sample and each column of X = one feature.

### Functions for you to fill in ###

# Given principal component vectors produced using
# pcs = principalComponents(X), 
# this function returns a new data array in which each sample in X 
# has been projected onto the first n_components principcal components.
def projectOntoPC(X, pcs, n_components):
    centeredX = centerData(X)
    #sortedPrincipalComponents = principalComponents(centeredX)
    V = pcs[:,0:n_components]
    #V = sortedPrincipalComponents[:,0:n_components] # get the first n_components cols
    return np.dot(centeredX, V)

# Returns a new dataset with features given by the mapping 
# which corresponds to the quadratic kernel.
def cubicFeatures(X):
    n, d = X.shape
    X_withones = np.ones((n,d+1))
    X_withones[:,:-1] = X
    new_d = int((d+1)*(d+2)*(d+3)/6)
    newData = np.zeros((n, new_d))

    for example in range(n):

        ctr = 0 # keep a running index
        variables = X_withones[example]

        # add all cubes
        for var in range(d+1):
            newData[example][ctr] = variables[var] ** 3
            ctr += 1

        # square the variable, multiply by every other variable (including 1)
        for var in range(d+1):
            for other_var in range(d+1):
                if not other_var == var: # if not same index
                    newData[example][ctr] = (3**0.5) * (variables[var] ** 2) * variables[other_var]
                    ctr+=1

        # add every distinct combo of three things
        for v1 in range(d+1):
            for v2 in range(v1, d+1):
                for v3 in range(v2, d+1):
                    if (v1 != v2 and v2 != v3 and v3 != v1):
                        newData[example][ctr] = (6**0.5) * variables[v1] * variables[v2] * variables[v3]
                        ctr += 1

    return newData


### Functions which are already complete, for you to use ###

# Returns a centered version of the data,
# where each feature now has mean = 0
def centerData(X):
    featureMeans = X.mean(axis = 0)
    return(X - featureMeans)

# Returns the principal component vectors of the data,
# sorted in decreasing order of eigenvalue magnitude.
def principalComponents(X):
    centeredData = centerData(X) # first center data
    scatterMatrix = np.dot(centeredData.transpose(), centeredData)
    eigenValues,eigenVectors = np.linalg.eig(scatterMatrix)
    # Re-order eigenvectors by eigenvalue magnitude: 
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenVectors

# Given the principal component vectors as the columns of matrix pcs,  
# this function projects each sample in X onto the first two principal components
# and produces a scatterplot where points are marked with the digit depicted in the corresponding image.
# labels = a numpy array containing the digits corresponding to each image in X.
def plotPC(X, pcs, labels):
    pc_data = projectOntoPC(X, pcs, n_components = 2)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:,0],pc_data[:,1], alpha=0, marker = ".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i,0],pc_data[i,1]))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show(block=True)

# Given the principal component vectors as the columns of matrix pcs,  
# this function reconstructs a single image 
# from its principal component representation, x_pca. 
# X = the original data to which PCA was applied to get pcs.
def reconstructPC(x_pca, pcs, n_components, X):
    featureMeans = X - centerData(X)
    featureMeans = featureMeans[0,:]
    x_reconstructed = np.dot(x_pca, pcs[:,range(n_components)].T) + featureMeans
    return x_reconstructed
