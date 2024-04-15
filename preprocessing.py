import matplotlib
import csv
import pandas as pd

def pickDataset(dataset):
    return {
    '1' : pd.read_csv("data\Signal1_drive1.csv"),
    '2' : pd.read_csv("data\Signal2_drive1.csv"),
    '3' : pd.read_csv("data\Signal3_drive1.csv")
    }[dataset]


                          
     