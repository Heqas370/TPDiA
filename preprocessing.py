import pandas as pd 
from sklearn.preprocessing import StandardScaler

def pickDataset(dataset):
    match dataset:
        case 0:
            ### Standard drive number 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1.csv")

        case 1:
            ### Standard drive number 2
            signal_1 =  pd.read_csv("data\Signal1_drive2.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive2.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive2.csv")

        case 2:
            ### Standard drive number 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1_case1.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1.csv")

        case 3:
            ### Anomaly signal 2 case 1 drive 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1_case1.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1.csv")

        case 4:
            ### Anomaly signal 2 case 2 drive 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1_case2.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1.csv")

        case 5:
            ### Anomaly signal 2 case 3 drive 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1_case3.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1.csv")

        case 6:
            ### Anomaly signal 2 case 1 drive 2
            signal_1 =  pd.read_csv("data\Signal1_drive2.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive2_case1.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive2.csv")
        
        case 7:
            ### Anomaly signal 2 case 2 drive 2
            signal_1 =  pd.read_csv("data\Signal1_drive2.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive2_case2.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive2.csv")

        case 8:
            ### Anomaly signal 2 case 3 drive 2
            signal_1 =  pd.read_csv("data\Signal1_drive2.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive2_case2.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive2.csv")

        case 9:
            ### Anomaly signal 3 case 1 drive 1
            signal_1 =  pd.read_csv("data\Signal1_drive1.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive1.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive1_case1.csv")

        case 10:
            ### Anomaly signal 2 case 3 drive 2
            signal_1 =  pd.read_csv("data\Signal1_drive2.csv"),
            signal_2 =  pd.read_csv("data\Signal2_drive2.csv"),
            signal_3 =  pd.read_csv("data\Signal3_drive2_case1.csv")

    return signal_1, signal_2, signal_3


def standiraze(signal_1, signal_2, signal_3):
    
    input_data = pd.concat((signal_1['BMS_Current'], signal_2['BMS_Voltage'], signal_3['BMS_SOC']), axis=1)

    scaler = StandardScaler()
    data = input_data.dropna()

    dataset = pd.Dataframe(scaler.fit_transform(data.values), columns = data.columns, index = data.index)

    return dataset, scaler

                          
     