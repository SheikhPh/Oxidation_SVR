import pandas as pd
import numpy as np
import random

FileName = 'Data ANN Oxidation'
Column_Names = ['Temperature', 'Nb', 'Mo', 'Cr','Al', 'Ti', 'Ta','Time', 'MC']
df = pd.read_excel(FileName + '.xlsx', sheet_name=0)

N_data = 140

Data = np.zeros((140, 9))

for i in range(N_data):
    for j in range(len(Column_Names)):
        Data[i][j] = df[Column_Names[j]][i]

random.seed(143)
Train_set = Data[np.random.choice(N_data, int(0.7 * N_data), replace=False)]
print(len(Data))
print(len(Train_set))
print(Train_set[1])
Test_set = []
ii = 0

for d in Data:
    flag = True
    for t in Train_set:
        if np.array_equal(d,t):
            flag = False
            ii+=1
            break
    if flag:
        Test_set.append(d)
print(ii)
print(len(Test_set))

c_coin = 0
for d in Data:
    flag = True
    for dd in Data:
        if np.array_equal(d, dd):
            c_coin+=1

c_coin -= N_data
print(c_coin)