import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def search_for_outliers(
        training_data: int,
        testing_data: int):

    signal_1, signal_2, signal_3 = preprocessing.pickDataset(training_data)
    signal_1_AN, signal_2_AN, signal_3_AN = preprocessing.pickDataset(testing_data)

    X_train = preprocessing.prepare_data(signal_1, signal_2, signal_3)
    X_test = preprocessing.prepare_data(signal_1_AN, signal_2_AN, signal_3_AN)

    clf = LocalOutlierFactor(contamination=0.0002, novelty=True, n_neighbors=500, metric='euclidean')

    clf.fit(X_train)

    y_pred = clf.predict(X_test)

    X_test['anomaly'] = y_pred

    anomaly = X_test.loc[X_test['anomaly']== -1]
    print(anomaly)

    plt.scatter(X_test.index, X_test['anomaly'], c=X_test['anomaly'], cmap='viridis')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomalies detected by Isolation Forest')
    plt.colorbar(label = 'Anoamly Score')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, X_test['BMS_Voltage'], label ='Voltage Data')
    plt.scatter(X_test.index, X_test['BMS_Voltage'], c=X_test['anomaly'], cmap='viridis', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
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
        clf = LocalOutlierFactor(contamination=0.01, novelty=True, n_neighbors=n, metric='euclidean')

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


if __name__ == "__main__":

    matplotlib.use('TKAgg')
    search_for_outliers(0, 6)