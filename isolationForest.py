import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import preprocessing
import matplotlib
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn import model_selection

def tune_parameters():
    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(0)
    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(2)

    X_train, _ = preprocessing.standarize(BMS_Current, BMS_Voltage, BMS_Soc)
    X_test, _ = preprocessing.standarize(BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)

    clf = IsolationForest()

    param_grid = {'random_state': list(range(50, 150, 50)),
                  'n_estimators': list(range(10, 150, 50)),
                  'random_state': list(range(10, 800, 100)),
                  'contamination': [0.003, 0.005, 0.01],
                  'bootstrap': [True, False]}
    
    grid_dt_estimator = model_selection.GridSearchCV(clf,param_grid,scoring='roc_auc',
                                                     refit= True, cv= 10, return_train_score= True,
                                                     verbose= 3, n_jobs= -1)
    
    grid_dt_estimator.fit(X_test, y= BMS_Voltage_AN['is_anomaly'])
    print(grid_dt_estimator.best_params_)

def search_for_outliers():

    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(1)
    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(1)

    X_train, _ = preprocessing.standarize(BMS_Current, BMS_Voltage, BMS_Soc)
    X_test, _ = preprocessing.standarize(BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)

    to_model_columns = X_train.columns[0:3]
    clf = IsolationForest(contamination= 0.002, random_state=10, n_estimators=7000, max_samples=5000, max_features=3,
                           bootstrap=True, verbose=2)
    clf.fit(X_train[to_model_columns])

    X_test['scores'] = clf.decision_function(X_test)
    y_pred = clf.predict(X_test[to_model_columns])
    X_test['anomaly'] = y_pred

    anomaly = X_test.loc[X_test['anomaly']== -1]
    print(anomaly)

    outliers = X_test.loc[X_test['anomaly']== -1]
    outlier_index = list(outliers.index)
    print(X_test['anomaly'].value_counts())

    X_test['anomaly'] = y_pred

    #Plotting anomalies
    plt.scatter(X_test.index, X_test['anomaly'], c=X_test['anomaly'], cmap='viridis')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomalies Detected by Isolation Forest')
    plt.colorbar(label= 'Anomaly Score')
    plt.show()

    #Plot timeseries data with anomalies

    plt.figure(figsize=(12,6))
    plt.plot(X_test.index, X_test['BMS_Voltage'], label='Voltage Data')
    plt.scatter(X_test.index, X_test['BMS_Voltage'], c=X_test['anomaly'], cmap='Viridis', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Voltage Time Series with Anomalies Detected by Isolation Forest')
    plt.colorbar(label='Anomaly Score')
    plt.legend()
    plt.show()

def test_with_metrics():
    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_score_results = []

    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(0)
    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(2)

    X_train, _ = preprocessing.standarize(BMS_Current, BMS_Voltage, BMS_Soc)
    X_test, _ = preprocessing.standarize(BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)

    for n in range(10, 200, 10):
        clf = IsolationForest(contamination=0.008, random_state= 300,
                              n_estimators= 5000, max_samples= n, max_features= 1, bootstrap= False, verbose=1)
        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        accuracy_results.append(accuracy_score(BMS_Voltage_AN)['isanomaly'], y_pred)
        precision_results.append(precision_score(BMS_Voltage_AN)['isanomaly'], y_pred)
        f1_score_results.append(f1_score(BMS_Voltage_AN['isanomaly'],y_pred))
        recall_results.append(recall_score(BMS_Voltage_AN['isanomaly'],y_pred))

    fig, (ax1, ax2, ax3, ax4) = plt.subplot(4, 1, figsize=(10, 12))
    fig.suptittle('Performance Metrics')

    ax1.plot(list(range(10, 200, 10)), accuracy_results)
    ax1.set_ylabel('Accuracy')

    ax2.plot(list(range(10, 200, 10)), precision_results)
    ax2.set_ylabel('Precision')

    ax3.plot(list(range(10, 200, 10)), f1_score_results)
    ax3.set_ylabel('F1 Score')

    ax4.plot(list(range(10, 200, 10)), recall_results)
    ax4.set_ylabel('Recall')
    ax4.set_xlabel('n_neighbours')

    plt.tight_layout()
    plt.show()