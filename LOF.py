import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import numpy as np



def search_for_outliers(
        training_data: int,
        testing_data: int):

    signal_1, signal_2, signal_3 = preprocessing.pickDataset(training_data)
    signal_1_AN, signal_2_AN, signal_3_AN = preprocessing.pickDataset(testing_data)

    X_train = preprocessing.prepare_data(signal_1, signal_2, signal_3)
    X_test = preprocessing.prepare_data(signal_1_AN, signal_2_AN, signal_3_AN)

    corr_train = preprocessing.row_corelation(pd.DataFrame(X_train['BMS_Current']), pd.DataFrame(X_train['BMS_Voltage']))
    corr_test = preprocessing.row_corelation(pd.DataFrame(X_test['BMS_Current']), pd.DataFrame(X_test['BMS_Voltage']))

    clf = LocalOutlierFactor(contamination=0.0007, novelty=True, n_neighbors=6000, metric='euclidean')

    clf.fit(corr_train)

    y_pred = clf.predict(corr_test)

    X_test['anomaly'] = y_pred

    anomaly = X_test.loc[X_test['anomaly']== -1]
    print(anomaly)

    outliers = X_test.loc[X_test['anomaly']==-1]
    
    X_test['anomaly'] = y_pred


    plt.scatter(X_test.index, X_test['anomaly'], c=X_test['anomaly'], cmap='viridis')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomalies detected by Isolation Forest')
    plt.colorbar(label = 'Anoamly Score')
    plt.show()

    Time = signal_2_AN['Time']
    anomaly = [i for i, value in enumerate(X_test['anomaly']) if value == -1]
    valueV = [X_test['BMS_Voltage'][i] for i in anomaly]
    timeV = [Time[i] for i in anomaly]

    plt.figure(figsize=(12, 6))
    plt.plot(Time, X_test['BMS_Voltage'], label ='Voltage Data')
    plt.plot(Time, X_test['BMS_Current'], label ='Current Data')
    plt.plot(Time, X_test['BMS_SOC'], label ='SOC Data')
    plt.scatter(timeV, valueV, c='r', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage, Current, SOC [V, A, %]')
    plt.colorbar(label='Anomaly Score')
    plt.legend(loc='lower left')
    plt.show()

    fpr, tpr, _ = roc_curve(signal_2_AN['is_anomaly'], y_pred)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05 ])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def test_with_metrics(
        training_data: int,
        testing_data: int,
        testing_range: list):

    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_score_results = []

    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(training_data)
    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(testing_data)

    X_train = preprocessing.prepare_data(BMS_Current, BMS_Voltage, BMS_Soc)
    X_test = preprocessing.prepare_data(BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)

    corr_train = preprocessing.row_corelation(pd.DataFrame(X_train['BMS_Current']), pd.DataFrame(X_train['BMS_Voltage']))
    corr_test = preprocessing.row_corelation(pd.DataFrame(X_test['BMS_Current']), pd.DataFrame(X_test['BMS_Voltage']))

    for n in range(testing_range[0], testing_range[1], testing_range[2]):
        clf = LocalOutlierFactor(contamination=0.01, novelty=True, n_neighbors=n, metric='euclidean')

        clf.fit(corr_train)
        y_pred = clf.predict(corr_test)
        
        y_true_minus1 = [BMS_Voltage_AN['is_anomaly'][i] for i in range(len(BMS_Voltage_AN['is_anomaly'])) if BMS_Voltage_AN['is_anomaly'][i] == -1]
        y_pred_minus1 = [y_pred[i] for i in range(len(y_pred)) if BMS_Voltage_AN['is_anomaly'][i] == -1]
        
        accuracy_results.append(accuracy_score(y_true_minus1, y_pred_minus1))
        precision_results.append(precision_score(y_true_minus1, y_pred_minus1))
        f1_score_results.append(f1_score(y_true_minus1,y_pred_minus1))
        recall_results.append(recall_score(y_true_minus1,y_pred_minus1))



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,12))
    fig.suptitle('Performance Metrics')

    ax1.plot(list(range(testing_range[0], testing_range[1], testing_range[2])), accuracy_results)
    ax1.set_ylabel('Accuracy')

    ax2.plot(list(range(testing_range[0], testing_range[1], testing_range[2])), precision_results)
    ax2.set_ylabel('Precision')

    ax3.plot(list(range(testing_range[0], testing_range[1], testing_range[2])), f1_score_results)
    ax3.set_ylabel('F1 Score')

    ax4.plot(list(range(testing_range[0], testing_range[1], testing_range[2])), recall_results)
    ax4.set_ylabel('Recall')
    ax4.set_xlabel('n_neighbours')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    matplotlib.use('TKAgg')
    # test_with_metrics(0, 3, [1000, 5000, 1000])
    search_for_outliers(1, 6)