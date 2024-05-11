from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
import feature_extraction
import visualization

def search_for_outliers(
        training_data: int,
        testing_data: int
):

    corr_train, corr_test, X_test, Time, y_true = feature_extraction.correlation_extraction(training_data, testing_data)

    clf = OneClassSVM(nu=1, kernel='rbf', gamma=0.01)
    clf.fit(corr_train)

    y_pred = clf.predict(corr_test)
    X_test['anomaly'] = y_pred

    anomaly = X_test.loc[X_test['anomaly']== -1]
    print(anomaly)

    print(X_test['anomaly'].value_counts())

    ### Plot detected anomalies
    visualization.plot_detected_anomalies(X_test)

    ### Plot anomalies
    visualization.plot_anomalies(X_test, Time)

    ### Plot ROC curve
    visualization.plot_ROC_curve(y_true, y_pred)


def test_with_metrics(
        training_data: int,
        testing_data: int,
        testing_range: list):

    f1_score_results = []

    corr_train, corr_test, _, _, y_true = feature_extraction.correlation_extraction(training_data, testing_data)


    for n in range(testing_range[0], testing_range[1], testing_range[2]):
        clf = OneClassSVM(nu=n, kernel='rbf', gamma=0.01)

        clf.fit(corr_train)
        y_pred = clf.predict(corr_test)
        f1_score_results.append(f1_score(y_true,y_pred))

    visualization.plot_F1_score(f1_score,[1000,2000,3000,4000,5000])


if __name__=="__main__":
    search_for_outliers(0,2)
    test_with_metrics()