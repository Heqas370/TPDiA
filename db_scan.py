import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import preprocessing
import matplotlib
from scipy.spatial.distance import seuclidean

def clasterization(epsVal, minSampleVal):

    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(1)

    X_test, _ = preprocessing.standarize(
        BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)
    
    k = 2 * X_test.shape[-1] - 1
    get_kdist_plot(X=X_test, k=k)

    dbscan = DBSCAN(eps=epsVal, min_samples=minSampleVal, metric="sqeuclidean")
    cluster_labels = dbscan.fit_predict(X_test)
    unique_labels = set(dbscan.labels_)
    for label in unique_labels:
        if label == -1:
            noise_points = X_test[cluster_labels == label]
    
    #VISUALIZATION
    _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(22,15))
    for label in unique_labels:
        cluster_points = X_test[dbscan.labels_ == label]
        ax.scatter(cluster_points['BMS_Current'], cluster_points['BMS_Voltage'],
                    cluster_points['BMS_SOC'], s=10, label=f'Cluster {label}')
    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Voltage [V]")
    ax.set_Zlabel("SOC [%]")

    ax.legend()
    plt.show()

    silhouette_avg = silhouette_score(X_test, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)

    return noise_points, X_test

def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):

    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    distances, _ = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    plt.figure(figsize=(8, 8))
    plt.plot(distances)
    plt.xlabel('Points/Object in dataset', fontsize=12)
    plt.ylabel('Sorted {} nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()

if __name__ == "__main__":
    matplotlib.use('TKAgg')
    nosie_points, trainingData = clasterization(0.45, 6)
    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(1)

    if not nosie_points.empty:
        plt.figure(figsize=(22,10))
        plt.subplot(221)
        plt.plot(BMS_Current['Time'], trainingData['BMS_Current'],label="Time Series Data", color='blue')
        for index in nosie_points.index:
            plt.scatter(BMS_Current.at[index, 'Time'], trainingData.iat[index,0], color='red', label='Highlighted Point')
        plt.title("Current Anomaly Dection")
        plt.grid(True)
        plt.subplot(222)
        plt.plot(BMS_Voltage['Time'], trainingData['BMS_Voltage'],label="Time Series Data", color='blue')
        for index in nosie_points.index:
            plt.scatter(BMS_Voltage.at[index, 'Time'], trainingData.iat[index,1], color='red', label='Highlighted Point')
        plt.title("Volage Anomaly Detection")
        plt.grid(True)
        plt.subplot(223)
        plt.plot(BMS_Soc['Time'], trainingData['BMS_Voltage'],label="Time Series Data", color='blue')
        for index in nosie_points.index:
            plt.scatter(BMS_Soc.at[index, 'Time'], trainingData.iat[index,2], color='red', label='Highlighted Point')
        plt.title("Soc Anomaly Detection")
        plt.grid(True)
        plt.show()
    else:
        print("No anomalies found")