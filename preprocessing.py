import pandas as pd 

def pickDataset(dataset):
    match dataset:
        case 0:
            ### standard drive number 1
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1.csv")

        case 1:
            ### standard drive number 2
            signal_1 =  pd.read_csv("data/signal1_drive2.csv")
            signal_2 =  pd.read_csv("data/signal2_drive2.csv")
            signal_3 =  pd.read_csv("data/signal3_drive2.csv")

        case 2:
            ### Anomaly signal 2 case 1 drive 1
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1_case1.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1.csv")

        case 3:
            ### Anomaly signal 2 case 2 drive 1
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1_case2.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1.csv")

        case 4:
            ### Anomaly signal 2 case 3 drive 1
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1_case2.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1.csv")

        case 5:
            ### Anomaly signal 2 case 1 drive 2
            signal_1 =  pd.read_csv("data/signal1_drive2.csv")
            signal_2 =  pd.read_csv("data/signal2_drive2_case1.csv")
            signal_3 =  pd.read_csv("data/signal3_drive2.csv")
        
        case 6:
            ### Anomaly signal 2 case 2 drive 2
            signal_1 =  pd.read_csv("data/signal1_drive2.csv")
            signal_2 =  pd.read_csv("data/signal2_drive2_case2.csv")
            signal_3 =  pd.read_csv("data/signal3_drive2.csv")

        case 7:
            ### Anomaly signal 2 case 3 drive 2
            signal_1 =  pd.read_csv("data/signal1_drive2.csv")
            signal_2 =  pd.read_csv("data/signal2_drive2_case3.csv")
            signal_3 =  pd.read_csv("data/signal3_drive2.csv")

        case 8:
            ### Anomaly signal 3 case 1 drive 1
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1_case1.csv")

        case 9:
            ### Anomaly signal 2 case 3 drive 2
            signal_1 =  pd.read_csv("data/signal1_drive1.csv")
            signal_2 =  pd.read_csv("data/signal2_drive1.csv")
            signal_3 =  pd.read_csv("data/signal3_drive1_case2.csv")

    return signal_1, signal_2, signal_3


def prepare_data(signal_1, signal_2, signal_3):
    
    input_data = pd.concat((signal_1['BMS_SOC'], signal_2['BMS_Voltage'], signal_3['BMS_Current']), axis=1)

    data = input_data.dropna()
    dataset = pd.DataFrame(data.values, columns = data.columns, index = data.index)

    return dataset

def row_corelation(df1,df2):

    signal_1 = abs(df1.values)
    signal_2 = abs(df2.values)
    result = signal_1 - signal_2

    result_df = pd.DataFrame(result,columns=['result_signal'])

    result_df.index = df1.index

    return result_df                  
     
def derivate(dataframe, drop_value):
    derivate = (dataframe.diff(drop_value) / dataframe.index_to_series().diff(drop_value).dt.total_seconds()).dropna()
    derivate.index = dataframe.index[0: -drop_value]

    return derivate