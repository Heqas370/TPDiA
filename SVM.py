import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import preprocessing
from sklearn import model_selection

def search_for_outliers(
        training_data: int,
        testing_data: int
):

    BMS_Current, BMS_Voltage, BMS_Soc = preprocessing.pickDataset(training_data)
    BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN = preprocessing.pickDataset(testing_data)

    X_train = preprocessing.prepare_data(BMS_Current, BMS_Voltage, BMS_Soc)
    X_test = preprocessing.prepare_data(BMS_Current_AN, BMS_Voltage_AN, BMS_Soc_AN)

    to_model_columns = X_train.columns[0:3]
    clf = OneClassSVM(nu=1, kernel='rbf', gamma=0.01)
    clf.fit(X_train[to_model_columns])

    X_test['scores'] = clf.decision_function(X_test)
    y_pred = clf.predict(X_test[to_model_columns])
    X_test['anomaly'] = y_pred

    anomaly = X_test.loc[X_test['anomaly']== -1]
    print(anomaly)

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
    plt.scatter(X_test.index, X_test['BMS_Voltage'], c=X_test['anomaly'], cmap='viridis', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Voltage Time Series with Anomalies Detected by Isolation Forest')
    plt.colorbar(label='Anomaly Score')
    plt.legend()
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

    for n in range(testing_range[0], testing_range[1], testing_range[2]):
        clf = OneClassSVM(nu=n, kernel='rbf', gamma=0.01)

        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        accuracy_results.append(accuracy_score(BMS_Voltage_AN)['isanomaly'], y_pred)
        precision_results.append(precision_score(BMS_Voltage_AN)['isanomaly'], y_pred)
        f1_score_results.append(f1_score(BMS_Voltage_AN['isanomaly'],y_pred))
        recall_results.append(recall_score(BMS_Voltage_AN['isanomaly'],y_pred))

    fig, (ax1, ax2, ax3, ax4) = plt.subplot(4, 1, figsize=(10, 12))
    fig.suptittle('Performance Metrics')

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


if __name__=="__main__":
    search_for_outliers(0,2)