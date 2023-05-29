from livelossplot import PlotLossesKeras
from tensorflow import keras
import numpy.random
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


FileName = 'Data ANN Oxidation del'
Column_Names = ['Temperature', 'Nb', 'Mo', 'Cr','Al', 'Ti', 'Ta','Time', 'MC']
df = pd.read_excel(FileName + '.xlsx', sheet_name=0)
N_column = len(Column_Names) #liczba danych wraz z wynikiem
N_data = len(df[Column_Names[0]]) #ilość danych
print(N_data)
Data = np.zeros((N_data, 9)) #dane wraz z wynikami

for i in range(N_data):
    for j in range(len(Column_Names)):
        Data[i][j] = df[Column_Names[j]][i]




Train_set = Data[np.random.choice(N_data, int(.8 * N_data), replace=False)]
Train_set_X = np.zeros((len(Train_set), N_column - 1))
Train_set_Y = np.zeros(len(Train_set))
print(len(Train_set))
for i in range(len(Train_set)):
    Train_set_X[i] = Train_set[i][:-1]
    Train_set_Y[i] = Train_set[i][-1]


Test_set = np.zeros((0, N_column))
Test_set_X = np.zeros((0, N_column - 1))
Test_set_Y = np.zeros(0)
ii = 0
for d in Data:
    flag = True
    for t in Train_set:
        if np.array_equal(d,t):
            flag = False
            break
    if flag:
        Test_set = np.vstack([Test_set, d])
        #Test_set = np.append(Test_set, [d], axis=0)
        Test_set_X = np.vstack([Test_set_X, d[:-1]])
        Test_set_Y = np.append(Test_set_Y, d[-1])



#standar scaling from SVM
sc_X = StandardScaler()
sc_Y = StandardScaler()
Train_set_X = sc_X.fit_transform(Train_set_X)
Train_set_Y = Train_set_Y.reshape(-1, 1)
Train_set_Y = sc_Y.fit_transform(Train_set_Y)

Test_set_X = sc_X.transform(Test_set_X)
Test_set_Y = Test_set_Y.reshape(-1, 1)
Test_set_Y = sc_Y.transform(Test_set_Y)





