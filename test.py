import pandas as pd

from DataWrangling import datawrangle
from LogisticRegression import logisticreg_fitmodel

train_data_file = 'data/TrainDataSet/train.csv'
test_data_file = 'data/TestDataSet/test.csv'
train_df = pd.read_csv(train_data_file)
test_df = pd.read_csv(test_data_file)
print('** test df ***', test_df.head())
x = datawrangle(train_df)
print('train data head',x.head())
y = datawrangle(test_df)
print('test data head',y.head())
acc, pred = logisticreg_fitmodel(x,y)
print(acc, pred)