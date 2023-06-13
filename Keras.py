from livelossplot import PlotLossesKeras
from tensorflow import keras
import numpy.random
import pandas as pd
import numpy as np


plotlosses = PlotLossesKeras()
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
layer= keras.layers.Normalization()
layer.adapt(Train_set)
Train_set = layer(Train_set)


Train_set_X = np.zeros((len(Train_set), N_column - 1))
Train_set_Y = np.zeros(len(Train_set))
print(len(Train_set))
for i in range(len(Train_set)):
    Train_set_X[i] = Train_set[i][:-1]
    Train_set_Y[i] = Train_set[i][-1]


Test_set = np.zeros((0, N_column))

ii = 0
for d in Data:
    flag = True
    for t in Train_set:
        if np.array_equal(d,t):
            flag = False
            break
    if flag:
        Test_set = np.vstack([Test_set, d])

Test_set_X = np.zeros((len(Test_set), N_column - 1))
Test_set_Y = np.zeros(len(Test_set))
Test_set = layer(Test_set)
for i in range(len(Test_set)):
    Test_set_X[i] = Test_set[i][:-1]
    Test_set_Y[i] = Test_set[i][-1]

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='relu', input_shape=[N_column-1]))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#50, 50, 10, 1 - .56

model.compile(
    loss='MeanSquaredError',
    optimizer=keras.optimizers.Adam())
model.summary()
model.fit(Train_set_X, Train_set_Y, epochs=250, batch_size=16)


#standar scaling from SVM
