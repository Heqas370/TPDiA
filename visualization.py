import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc


def plot_F1_score(f1_score, range, test_param):

    plt.figure()
    plt.title("Performance Metric")
    plt.plot(range, f1_score)
    plt.ylabel('F1 score')
    plt.xlabel(test_param)

    plt.tight_layout()
    plt.show()

def plot_ROC_curve(y_true, y_pred):

    fpr,tpr,_ = roc_curve(y_true, y_pred)
    roc_auc =auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)'% roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def plot_anomalies(X_test, Time):

    anomaly = [i for i, value in enumerate(X_test['anomaly']) if value == -1]
    value_AN = [X_test['BMS_Voltage'][i] for i in anomaly]
    time_AN = [Time[i] for i in anomaly]

    plt.figure(figsize=(12, 6))
    plt.plot(Time, X_test['BMS_Current'], label='Current Data')
    plt.plot(Time, X_test['BMS_Voltage'], label='Voltage Data')
    plt.plot(Time, X_test['BMS_SOC'], label='SOC Data')
    plt.scatter(time_AN, value_AN, c='r', label='anomalies')
    plt.xlabel('Time')
    plt.ylabel('Voltage, Current, SOC')
    plt.legend(loc='lower left')
    plt.show()

def plot_detected_anomalies(X_test):
    plt.scatter(X_test.index, X_test['anomaly'], c=X_test['anomaly'], cmap='viridis')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    plt.title('Detected anomalies')
    plt.colorbar(label = 'Anomaly Score')
    plt.show()