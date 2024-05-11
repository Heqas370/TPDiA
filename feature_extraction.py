import preprocessing
import pandas as pd
import numpy as np

def correlation_extraction(training_data, testing_data):

    signal_1, signal_2, signal_3 = preprocessing.pickDataset(training_data)
    signal_1_AN, signal_2_AN, signal_3_AN = preprocessing.pickDataset(testing_data)

    X_train = preprocessing.prepare_data(signal_1, signal_2, signal_3)
    X_test = preprocessing.prepare_data(signal_1_AN, signal_2_AN, signal_3_AN)

    corr_train = preprocessing.row_corelation(pd.DataFrame(X_train['BMS_Current']), pd.DataFrame(X_train['BMS_Voltage']))
    corr_test = preprocessing.row_corelation(pd.DataFrame(X_test['BMS_Current']), pd.DataFrame(X_test['BMS_Voltage']))

    Time = signal_2_AN['Time']
    y_true = signal_2_AN['is_anomaly']

    return corr_train, corr_test, X_test, Time, y_true


def slope_extraction(dataset):
    
    _, signal_2, signal_3 = preprocessing.pickDataset(dataset)


    signal_3['Time'] = pd.to_datetime(signal_3.Time, unit='s')
    signal_3.set_index(signal_3['Time'], inplace=True)
    derivate_signal3 = preprocessing.derivate(signal_3['BMS_SOC'])

    signal_2.set_index(signal_3['Time'], inplace=True)
    derivate_signal2 = preprocessing.derivate(signal_2['BMS_Voltage'])

    slope = derivate_signal3 / derivate_signal2
    result =pd.DataFrame({'slope': slope.values})
    result = result.fillna(0)
    result.replace([np.inf, -np.inf], 0, inplace=True)

    return result


def feature_slope(training_data, testing_data):

    slope_train = slope_extraction(training_data)
    slope_test = slope_extraction(testing_data)

    return slope_train, slope_test