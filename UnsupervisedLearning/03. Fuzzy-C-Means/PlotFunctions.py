# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 00:24:23 2021

@author: 34123
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal


def plot_random_init_iris_sepal(df_full):
    sepal_df = df_full.iloc[:,0:2]
    sepal_df = np.array(sepal_df)
    
    m1 = random.choice(sepal_df)
    m2 = random.choice(sepal_df)
    m3 = random.choice(sepal_df)

    cov1 = np.cov(np.transpose(sepal_df))
    cov2 = np.cov(np.transpose(sepal_df))
    cov3 = np.cov(np.transpose(sepal_df))
    
    x1 = np.linspace(4,8,150)  
    x2 = np.linspace(1.5,4.5,150)
    X, Y = np.meshgrid(x1,x2) 

    Z1 = multivariate_normal(m1, cov1)  
    Z2 = multivariate_normal(m2, cov2)
    Z3 = multivariate_normal(m3, cov3)
    
    # a new array of given shape and type, without initializing entries
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y   

    plt.figure(figsize=(10,10))
    plt.scatter(sepal_df[:,0], sepal_df[:,1], marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
    plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
    # making both the axis equal
    plt.axis('equal')                                                                 
    plt.xlabel('Sepal Length', fontsize=16)
    plt.ylabel('Sepal Width', fontsize=16)
    plt.title('Initial Random Clusters(Sepal)', fontsize=22)
    plt.grid()
    plt.show()
    

def plot_random_init_iris_petal(df_full):
    petal_df = df_full.iloc[:,2:4]
    petal_df = np.array(petal_df)
    
    m1 = random.choice(petal_df)
    m2 = random.choice(petal_df)
    m3 = random.choice(petal_df)
    cov1 = np.cov(np.transpose(petal_df))
    cov2 = np.cov(np.transpose(petal_df))
    cov3 = np.cov(np.transpose(petal_df))

    x1 = np.linspace(-1,7,150)
    x2 = np.linspace(-1,4,150)
    X, Y = np.meshgrid(x1,x2) 

    Z1 = multivariate_normal(m1, cov1)  
    Z2 = multivariate_normal(m2, cov2)
    Z3 = multivariate_normal(m3, cov3)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y   

    plt.figure(figsize=(10,10))
    plt.scatter(petal_df[:,0], petal_df[:,1], marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
    plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5) 
    plt.axis('equal') 
    plt.xlabel('Petal Length', fontsize=16) 
    plt.ylabel('Petal Width', fontsize=16)
    plt.title('Initial Random Clusters(Petal)', fontsize=22)
    plt.grid()
    plt.show()
    
    
def plot_cluster_iris_sepal(df_full, labels, centers):
    # finding mode
    seto = max(set(labels[0:50]), key=labels[0:50].count) # 2
    vers = max(set(labels[50:100]), key=labels[50:100].count) # 1
    virg = max(set(labels[100:]), key=labels[100:].count) # 0
    
    # sepal
    s_mean_clus1 = np.array([centers[seto][0],centers[seto][1]])
    s_mean_clus2 = np.array([centers[vers][0],centers[vers][1]])
    s_mean_clus3 = np.array([centers[virg][0],centers[virg][1]])
    
    values = np.array(labels) #label

    # search all 3 species
    searchval_seto = seto
    searchval_vers = vers
    searchval_virg = virg

    # index of all 3 species
    ii_seto = np.where(values == searchval_seto)[0]
    ii_vers = np.where(values == searchval_vers)[0]
    ii_virg = np.where(values == searchval_virg)[0]
    ind_seto = list(ii_seto)
    ind_vers = list(ii_vers)
    ind_virg = list(ii_virg)
    
    sepal_df = df_full.iloc[:,0:2]
    
    seto_df = sepal_df[sepal_df.index.isin(ind_seto)]
    vers_df = sepal_df[sepal_df.index.isin(ind_vers)]
    virg_df = sepal_df[sepal_df.index.isin(ind_virg)]
    
    cov_seto = np.cov(np.transpose(np.array(seto_df)))
    cov_vers = np.cov(np.transpose(np.array(vers_df)))
    cov_virg = np.cov(np.transpose(np.array(virg_df)))
    
    sepal_df = np.array(sepal_df)
    
    x1 = np.linspace(4,8,150)  
    x2 = np.linspace(1.5,4.5,150)
    X, Y = np.meshgrid(x1,x2) 

    Z1 = multivariate_normal(s_mean_clus1, cov_seto)  
    Z2 = multivariate_normal(s_mean_clus2, cov_vers)
    Z3 = multivariate_normal(s_mean_clus3, cov_virg)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y   

    plt.figure(figsize=(10,10))                                                          
    plt.scatter(sepal_df[:,0], sepal_df[:,1], marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
    plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5) 
    plt.axis('equal')                                                                  
    plt.xlabel('Sepal Length', fontsize=16)
    plt.ylabel('Sepal Width', fontsize=16)
    plt.title('Final Clusters(Sepal)', fontsize=22)  
    plt.grid()
    plt.show()
    
    
    
def plot_cluster_iris_petal(df_full, labels, centers):
    # petal
    p_mean_clus1 = np.array([centers[seto][2],centers[seto][3]])
    p_mean_clus2 = np.array([centers[vers][2],centers[vers][3]])
    p_mean_clus3 = np.array([centers[virg][2],centers[virg][3]])
    
    petal_df = df_full.iloc[:,2:4]
    
    seto_df = petal_df[petal_df.index.isin(ind_seto)]
    vers_df = petal_df[petal_df.index.isin(ind_vers)]
    virg_df = petal_df[petal_df.index.isin(ind_virg)]
    
    cov_seto = np.cov(np.transpose(np.array(seto_df)))
    cov_vers = np.cov(np.transpose(np.array(vers_df)))
    cov_virg = np.cov(np.transpose(np.array(virg_df)))
    
    petal_df = np.array(petal_df) 
    
    x1 = np.linspace(0.5,7,150)  
    x2 = np.linspace(-1,4,150)
    X, Y = np.meshgrid(x1,x2) 

    Z1 = multivariate_normal(p_mean_clus1, cov_seto)  
    Z2 = multivariate_normal(p_mean_clus2, cov_vers)
    Z3 = multivariate_normal(p_mean_clus3, cov_virg)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y   

    plt.figure(figsize=(10,10))                                                         
    plt.scatter(petal_df[:,0], petal_df[:,1], marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
    plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5) 
    plt.axis('equal')                                               
    plt.xlabel('Petal Length', fontsize=16)
    plt.ylabel('Petal Width', fontsize=16)
    plt.title('Final Clusters(Petal)', fontsize=22)
    plt.grid()
    plt.show()
