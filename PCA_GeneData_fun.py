import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PCA_fun(gene_data, meta_data, no_of_features = 5000):
    
    ## Loading the data

    gene_df = pd.read_csv(gene_data)

    meta_df = pd.read_csv(meta_data)

    # print("data Shape : {}".format(gene_df.shape))



    ## Cleaning the data and getting the data in proper format

    # discarding the unnecessary columns
    X = gene_df.iloc[:, 2:32]

    # cleaning the data
    X = X.replace(['hhhh', 'ssssss'], np.nan)
    X = X.dropna()

    # for samples in rows and genes in columns format
    X = X.T
    
    if no_of_features == 'all':
        X = X.iloc[:, :]
    
    # for small no. of features(genes) as my PC is getting stuck
    else:
        X = X.iloc[:, :no_of_features]



    #print("data Shape : {}".format(X.shape))



    ## Preparing the class labels

    y = meta_df.iloc[:, 1]



    #3 Standardizing the data

    from sklearn.preprocessing import StandardScaler

    X_std = StandardScaler().fit_transform(X)



    ## Claculating the covariance matrix used for Eigen decomposition

    cov_mat = np.cov(X_std.T)

    #print('Covariance matrix \n%s' %cov_mat)



    ## Calculating Eigenvalues and Eigenvectors by doing Eigen decomposition of Covariance matrix

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    #print('Eigenvectors \n%s' %eig_vecs)
    #print('\nEigenvalues \n%s' %eig_vals)



    ## Sorting Eigenpairs

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    #print('Eigenvalues in descending order:')
    #for i in eig_pairs:
        #print(i[0])



    ## Creating Projection matrix by concatenating top 2 Eigenvectors

    matrix_w = np.hstack((eig_pairs[0][1].reshape(no_of_features,1),
                          eig_pairs[1][1].reshape(no_of_features,1)))

    #print('Matrix W:\n', matrix_w)
    #print('\nMatrix W shape:{}' .format(matrix_w.shape))



    ## Projecting our samples into the new Feature subspace

    Y = X_std.dot(matrix_w)



    ## Plotting the PC1 and PC2

    plt.figure(figsize=(10, 6))

    for lab in ('9', '7', '5', '0', '4', '6', '8', '10', '12', '11'):

        plt.scatter(Y[y==int(lab), 0],
                    Y[y==int(lab), 1],
                    label=lab)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig('gene_PCA.png')
    plt.show()

