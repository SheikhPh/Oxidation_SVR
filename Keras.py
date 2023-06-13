from livelossplot import PlotLossesKeras
from tensorflow import keras
import numpy.random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


best_model = None
avg_loss = 0
best_loss = 100
worst_loss = 0
for i in range(100):

    Train_set, Test_set = train_test_split(Data, train_size=0.8)
    layer = keras.layers.Normalization()
    layer.adapt(Train_set)
    Train_set = layer(Train_set)
    Test_set = layer(Test_set)


    Train_set_X = np.zeros((len(Train_set), N_column - 1))
    Train_set_Y = np.zeros(len(Train_set))
    for i in range(len(Train_set)):
        Train_set_X[i] = Train_set[i][:-1]
        Train_set_Y[i] = Train_set[i][-1]


    Test_set_X = np.zeros((len(Test_set), N_column - 1))
    Test_set_Y = np.zeros(len(Test_set))
    for i in range(len(Test_set)):
        Test_set_X[i] = Test_set[i][:-1]
        Test_set_Y[i] = Test_set[i][-1]


    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_shape=(8,)))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    #model.summary()
    model.fit(Train_set_X, Train_set_Y, epochs=250, batch_size=16, callbacks=None, verbose=0)

    l = model.evaluate(x = Test_set_X, y = Test_set_Y)
    l = l[0]
    if l > worst_loss:
        worst_loss = l
    if l < best_loss:
        best_loss = l
        best_model = model
    avg_loss = (avg_loss * i + l)/(i+1)

print(best_loss)
print(worst_loss)
print(avg_loss)