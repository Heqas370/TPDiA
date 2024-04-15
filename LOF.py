import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
import preprocessing

def search_for_outliers():

    signal_1, signal_2, signal_3 = preprocessing.pickDataset()
    signal_1_AN, signal_2_AN, signal_3_AN = preprocessing.pickDataset()

    X_train, _ = preprocessing.standarize(signal_1, signal_2, signal_3)
    X_test, _ = preprocessing.standarize(signal_1_AN, signal_2_AN, signal_3_AN)

    clf = LocalOutlierFactor(contamination=0.01, novelty=True, n_neighbors=250, metric='num_euclidean')

    clf.fit(X_train)

    y_pred = clf.predict(X_test)

    X_test['anomaly'] = y_pred
    outliers = X_test.locp[X_test['anomaly'] == -1 ]
    outlier_index = list(outliers.index)
    print(X_test['anomaly'].value_counts)

    X_test['anomaly'] = y_pred

    plt.scatter(X_test.index, X_test['anomaly'], c=X_test['anomaly'], cmap='viridis')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomalies detected by Isolation Forest')
    plt.colorbar(label = 'Anoamly Score')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, signal_1['BMS_Voltage'], label ='Voltage Data')
    plt.scatter(X_test.index, X_test['BMS_Voltage'], c=X_test['anomaly'], cmap='virdis', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.colorbar(label='Anomaly Score')
    plt.legend()
    plt.show()

def test_with_metrics():

    signal_1, signal_2, signal_3 = preprocessing.pickDataset()
    signal_1_AN, signal_2_AN, signal_3_AN = preprocessing.pickDataset()

    X_train, _ = preprocessing.standarize(signal_1, signal_2, signal_3)
    X_test, _ = preprocessing.standarize(signal_1_AN, signal_2_AN, signal_3_AN)

    contamination = 1/len(X_train)

    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_score_results = []

    for n in range(10, 1010, 50):
        clf = LocalOutlierFactor(contamination=contamination, novelty=True, n_neighbors=n)
        clf.fit(X_train)
        y_pred = clf.predict(X_test)

        accuracy_results.append(accuracy_score(signal_1_AN['is_anomaly'], y_pred))
        precision_results.append(precision_score(signal_1_AN['is_anomaly'], y_pred))
        recall_results.append(recall_score(signal_1_AN['is_anomaly'], y_pred))
        f1_score_results.append(f1_score(signal_1_AN['is_anomaly'], y_pred))

    fig, (ax1, ax2, ax3, ax4) = plt.subplot(4, 1, figsize=(10, 12))
    fig.suptittle('Performance Metrics')

    ax1.plot(list(range(10, 1010, 50)), accuracy_results)
    ax1.set_ylabel('Accuracy')

    ax2.plot(list(range(10, 1010, 50)), precision_results)
    ax2.set_ylabel('Precision')

    ax3.plot(list(range(10, 1010, 50)), recall_results)
    ax3.set_ylabel('Accuracy')

    ax4.plot(list(range(10, 1010, 50)), f1_score_results)
    ax4.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    matplotlib.use('TKAgg')

    search_for_outliers()