import csv
import pandas as pd

with open('C:/Users/yuiichiiros/Downloads/full_db_mask_norm3_PCA40_folds.csv') as csvfile:
    readCSV = pd.read_csv(csvfile)

dataset = readCSV.values

print(dataset)
