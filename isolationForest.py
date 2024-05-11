from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import preprocessing
from sklearn import model_selection
import visualization
import feature_extraction

def tune_parameters(
        training_data: int,
        testing_data: int):

    corr_train, corr_test, X_test, Time, y_true = feature_extraction.correlation_extraction(training_data, testing_data)

    clf = IsolationForest()

    param_grid = {'random_state': list(range(50, 150, 50)),
                  'n_estimators': list(range(10, 150, 50)),
                  'random_state': list(range(10, 800, 100)),
                  'contamination': [0.003, 0.005, 0.01],
                  'bootstrap': [True, False]}
    
    grid_dt_estimator = model_selection.GridSearchCV(clf,param_grid,scoring='roc_auc',
                                                     refit= True, cv= 10, return_train_score= True,
                                                     verbose= 3, n_jobs= -1)
    
    grid_dt_estimator.fit(corr_test, y=y_true)
    print(grid_dt_estimator.best_params_)

def search_for_outliers(
        training_data: int,
        testing_data: int
):

    corr_train, corr_test, X_test, Time, y_true = feature_extraction.correlation_extraction(training_data, testing_data)

    ### Training and testing
    clf = IsolationForest(contamination= 0.002, random_state=10, n_estimators=7000, max_samples=5000, max_features=3,
                           bootstrap=True, verbose=2)

    clf.fit(corr_train)
    X_test['scores'] = clf.decision_function(corr_test)
    y_pred = clf.predict(X_test)
    X_test['anomaly'] = y_pred
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
        clf = IsolationForest(contamination=0.008, random_state= 300,
                              n_estimators= 5000, max_samples= n, max_features= 1, bootstrap= False, verbose=1)
        clf.fit(corr_train)
        y_pred = clf.predict(corr_test)

        f1_score_results.append(f1_score(y_true,y_pred))

    visualization.plot_F1_score(f1_score,[1000,2000,3000,4000,5000])

if __name__=="__main__":
    search_for_outliers(0,3)