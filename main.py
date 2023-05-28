import pandas as pd
import numpy as np
import random
from sklearn.svm import SVR
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




#tworzenie setu testowego i trainowego
#np.random.seed(7)
#Dla 5: Dużo - 3, mało - 7
#Dla 10: Max: ~.97, Min: -.5, często >.9
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



#dopasowanie SVR
sc_X = StandardScaler()
sc_Y = StandardScaler()
Train_set_X = sc_X.fit_transform(Train_set_X)
Train_set_Y = Train_set_Y.reshape(-1, 1)
Train_set_Y = sc_Y.fit_transform(Train_set_Y)

Test_set_X = sc_X.transform(Test_set_X)
Test_set_Y = Test_set_Y.reshape(-1, 1)
Test_set_Y = sc_Y.transform(Test_set_Y)

svr = SVR(kernel='poly', degree = 10, epsilon=0.1)
svr.fit(Train_set_X, Train_set_Y)
pred_Y = svr.predict(Test_set_X)
print(svr.score(Train_set_X, Train_set_Y))
print(svr.score(Test_set_X, Test_set_Y))

# for i in range(len(Test_set_X)):
#      print(pred_Y[i], Test_set_Y[i])

#zliczanie takich samych wierszy
'''
c_coin = 0
for i in range(N_data):

    flag = True
    for j in range(i+1, N_data):
        if np.array_equal(Data[i][:-1], Data[j][:-1]):
            c_coin+=1
            print(i+2, j+2)

print(c_coin)
'''