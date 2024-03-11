#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Supervised K-ISOMAP for dimensionality reduction based metric learning

Python code to reproduce the results of the paper

"""

# Imports
import sys
import time
import warnings
import umap
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    return novos_dados

# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
def SupervisedPCA(dados, labels, d):
    dados = dados.T
    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features
    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1
    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    return novos_dados

# SK-ISOMAP implementation
def SK_Isomap(dados, k, d, target):
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the tangent spaces
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            matriz_pcs[i, :, :] = Wpca
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                if target[i] == target[j]:
                    B[i, j] = min(delta)
                else:
                    B[i, j] = sum(delta)
    # Computes geodesic distances in B
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])  
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    return output

# Train and test eight different supervised classifiers
def Classification(dados, target, method):
    lista = []
    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real, target, test_size=.5, random_state=42)
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train) 
    pred = neigh.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    pred = svm.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    pred = qda.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    mpl.fit(X_train, y_train)
    pred = mpl.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    pred = gpc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados, target, metric='euclidean')
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)
    print()
    print('Maximum balanced accuracy for %s features: %f' %(method, maximo))
    print()
    return [sc, average, maximo]

# Plot the scatterplots dor the 2D output data
def PlotaDados(dados, labels, metodo):
    nclass = len(np.unique(labels))
    if metodo == 'LDA':
        if nclass == 2:
            return -1
    # Encode the labels as integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to numbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    rotulos = np.array(rotulos)
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']
    plt.figure(1)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()

#%%%%%%%%%%%%%%%%%%%%  Data loading

# OpenML datasets
X = skdata.load_digits()                                           
#X = skdata.fetch_openml(name='car-evaluation', version=1)         
#X = skdata.fetch_openml(name='wine-quality-white', version=1)     
#X = skdata.fetch_openml(name='wine-quality-red', version=1)       
#X = skdata.fetch_openml(name='glass', version=1)                  
#X = skdata.fetch_openml(name='ecoli', version=1)                  
#X = skdata.fetch_openml(name='vowel', version=2)                  
#X = skdata.fetch_openml(name='collins', version=4)                
#X = skdata.fetch_openml(name='energy-efficiency', version=1)      
#X = skdata.fetch_openml(name='satimage', version=1)               
#X = skdata.fetch_openml(name='led24', version=1)                  
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)            
#X = skdata.fetch_openml(name='balance-scale', version=1)           
#X = skdata.fetch_openml(name='diabetes', version=1)                
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)          
#X = skdata.fetch_openml(name='grub-damage', version=2)             
#X = skdata.fetch_openml(name='banknote-authentication', version=1) 
#X = skdata.fetch_openml(name='vehicle', version=1)              
#X = skdata.fetch_openml(name='ionosphere', version=1)           
#X = skdata.fetch_openml(name='wall-robot-navigation', version=1)    
#X = skdata.fetch_openml(name='CIFAR_10_small', version=1)           # 30 PC's and 20% of samples
#X = skdata.fetch_openml(name='pendigits', version=1)                # 20% of samples
#X = skdata.fetch_openml(name='artificial-characters', version=1)    # 25% of samples
#X = skdata.fetch_openml(name='waveform-5000', version=1)            
#X = skdata.fetch_openml(name='nursery', version=1)                  # 25% of the samples
#X = skdata.fetch_openml(name='eye_movements', version=1)            # 30% of samples
#X = skdata.fetch_openml(name='zoo', version=1)                      
#X = skdata.fetch_openml(name='thyroid-dis', version=1)              
#X = skdata.fetch_openml(name='one-hundred-plants-shape', version=1) 
#X = skdata.fetch_openml(name='MNIST_784', version=1)                # 50 PC's and 5% of the samples

dados = X['data']
target = X['target']  

if 'details' in X.keys():
    if X['details']['name'] == 'CIFAR_10_small':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
        dados = myPCA(dados, 30).real.T
    elif X['details']['name'] == 'pendigits':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
    elif X['details']['name'] == 'artificial-characters':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    elif X['details']['name'] == 'nursery':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)            
    elif X['details']['name'] == 'eye_movements':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.3, random_state=42)
    elif X['details']['name'] == 'mnist_784':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
        dados = myPCA(dados, 50).real.T


n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

# Treat categorical features
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

# Number of neighbors
nn = round(sqrt(n))     # Número de vizinhos = raiz quadrada de n
print()
print('Number of samples (n): ', n)
print('Number of features (m): ', m)
print('Number of classes (c): ', c)
print('Number of Neighbors in k-NN graph (k): ', nn)
print()
print('Press enter to continue...')
input()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%%%% PLS
model = PLSRegression(n_components=2)
dados_pls = model.fit_transform(dados, y=target)

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados, y=target)

#%%%%%%%%%%%%% Supervised PCA
dados_suppca = SupervisedPCA(dados, target, 2)
dados_suppca = dados_suppca.T

#%%%%%%%%%%%% LDA
if c > 2:
    model = LinearDiscriminantAnalysis(n_components=2)
else:
    model = LinearDiscriminantAnalysis(n_components=1)
dados_lda = model.fit_transform(dados, target)

#%%%%%%%%%%% Supervised classification
L_umap = Classification(dados_pls[0], target, 'PLS')
L_umap = Classification(dados_umap, target, 'S-UMAP')
L_suppca = Classification(dados_suppca.real, target, 'SUP PCA')
L_lda = Classification(dados_lda, target, 'LDA')

PlotaDados(dados_pls[0], target, 'PLS')
PlotaDados(dados_umap, target, 'S-UMAP')
PlotaDados(dados_suppca, target, 'SUP PCA')
PlotaDados(dados_lda, target, 'LDA')

# SK-ISOMAP
dados_skiso = SK_Isomap(dados, nn, 2, target)
L_kiso = Classification(dados_skiso, target, 'SUP K-ISO')
PlotaDados(dados_skiso, target, 'SUP K-ISO')