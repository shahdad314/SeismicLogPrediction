#!/usr/bin/env python
# coding: utf-8


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import callbacks
import pandas as pd
from error_calc import *

### open the data file
dataset = np.loadtxt("train.csv", delimiter=",").astype('float32')

df = pd.DataFrame(dataset)
maxdata = (df.max()).values
mindata = df.min().values
meandata = df.mean(axis=0).values

ren1 = 300
y = range(ren1)
log_title = ['CNC', 'DT24', 'GR', 'RS', 'RD', 'ZDNC']
fig, axs = plt.subplots(1, 6, figsize=(15, 5))
fig.subplots_adjust(hspace=.5, wspace=.5)
axs = axs.ravel()
for i in range(6):
    axs[i].plot(dataset[:ren1, i], y)
    axs[i].invert_yaxis()
    axs[i].title.set_text(log_title[i])

plt.show()

net_dimension = dataset.shape[1] - 1


def normalisation(dataset, mindata, maxdata, meandata):
    normal_data = dataset - meandata
    normal_data /= (maxdata - mindata)
    return normal_data


def un_normalisation(normal_data, mindata, maxdata, meandata, label=False):
    if label == False:
        predict_data = normal_data * (maxdata - mindata)
        predict_data += meandata
    else:
        predict_data = normal_data * (maxdata[-1] - mindata[-1])
        predict_data += meandata[-1]
    return predict_data


normal_data = normalisation(dataset, mindata, maxdata, meandata)
xTrain, xTest, yTrain, yTest = train_test_split(normal_data[:, :-1], normal_data[:, -1], test_size=0.3, random_state=0)


early_stop = callbacks.EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=0, patience=7, verbose=1,
                                     mode='auto')

## creating a multi layered neural networks

model = Sequential()

# 1
model.add(Dense(10, input_dim=net_dimension))
model.add(Activation(activation='linear'))
# 2
model.add(Dense(40, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='elu'))
# 3
model.add(Dense(60, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='elu'))
# 4
model.add(Dense(100, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='linear'))
# 5
model.add(Dense(60, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='elu'))
# 6
model.add(Dense(30, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='elu'))
# 7
model.add(Dense(10, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='elu'))
# 8
model.add(Dense(5, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='linear'))
# 9
model.add(Dense(2, kernel_initializer='glorot_uniform'))
model.add(Activation(activation='linear'))
# 10
model.add(Dense(1))
model.add(Activation(activation='elu'))

model.compile(optimizer='adamax', loss='mse', metrics=['mape'])

test1 = normalisation(np.loadtxt("test1.csv", delimiter=","), mindata, maxdata, meandata)
test2 = normalisation(np.loadtxt("test2.csv", delimiter=","), mindata, maxdata, meandata)
test3 = normalisation(np.loadtxt("test3.csv", delimiter=","), mindata, maxdata, meandata)
test4 = normalisation(np.loadtxt("test4.csv", delimiter=","), mindata, maxdata, meandata)

### spliting the data from labels for the test sets
xtest1 = test1[:, :-1]
ytest1 = test1[:, -1]

xtest2 = test2[:, :-1]
ytest2 = test2[:, -1]

xtest3 = test3[:, :-1]
ytest3 = test3[:, -1]

xtest4 = test4[:, :-1]
ytest4 = test4[:, -1]


with tensorflow.device('/gpu:0'):
    history = model.fit(xTrain, yTrain, validation_split=0.8, epochs=100, batch_size=256, verbose=1,
                        callbacks=[early_stop])

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(xTrain, yTrain, verbose=0)
print("After Training:", (model.metrics_names, score))
score2 = model.evaluate(xTest, yTest, verbose=0)
print("After Training test:", (model.metrics_names, score2))

predict_label = un_normalisation(model.predict(xTest), mindata, maxdata, meandata, label=True)
measured_label = un_normalisation(yTest, mindata, maxdata, meandata, label=True)


# visualising the data
mse, rmse, mae, mape, r2 = error_calc(measured_label, predict_label)
ren1 = 500
rent1 = len(measured_label)
y = range(ren1)
plt.figure(figsize=(5, 15))
plt.plot(measured_label[:ren1], y, label='measured value')
plt.plot(predict_label[:ren1], y, label='predicted value')
# plt.invert_yaxis()
plt.title('ZDNC predicted values vs measured values for test set')
plt.legend()
plt.text(1.3, 0, 'mse  = {:,.4f}'.format(mse))
plt.text(1.3, 10, 'rmse = {:,.4f}'.format(rmse))
plt.text(1.3, 20, 'mae  = {:,.4f}'.format(mae))
plt.text(1.3, 30, 'mape= {:,.4f}'.format(mape))
plt.text(1.3, 40, 'r2      = {:,.4f}'.format(r2))

plt.show()

predict_test1 = un_normalisation(model.predict(xtest1), mindata, maxdata, meandata, label=True)
predict_test2 = un_normalisation(model.predict(xtest2), mindata, maxdata, meandata, label=True)
predict_test3 = un_normalisation(model.predict(xtest3), mindata, maxdata, meandata, label=True)
predict_test4 = un_normalisation(model.predict(xtest4), mindata, maxdata, meandata, label=True)

ytest1 = un_normalisation(ytest1, mindata, maxdata, meandata, label=True)
ytest2 = un_normalisation(ytest2, mindata, maxdata, meandata, label=True)
ytest3 = un_normalisation(ytest3, mindata, maxdata, meandata, label=True)
ytest4 = un_normalisation(ytest4, mindata, maxdata, meandata, label=True)

mse1, rmse1, mae1, mape1, r21 = error_calc(ytest1, predict_test1)
mse2, rmse2, mae2, mape2, r22 = error_calc(ytest2, predict_test2)
mse3, rmse3, mae3, mape3, r23 = error_calc(ytest3, predict_test3)
mse4, rmse4, mae4, mape4, r24 = error_calc(ytest4, predict_test4)


# visualization for four test sets

ren1 = 300

ren1 = len(ytest1)
ren2 = len(ytest2)
ren3 = len(ytest3)
ren4 = len(ytest4)

y1 = range(ren1)
y2 = range(ren2)
y3 = range(ren3)
y4 = range(ren4)

f = plt.figure(figsize=(15, 15))
ax1 = f.add_subplot(141)
ax2 = f.add_subplot(142)
ax3 = f.add_subplot(143)
ax4 = f.add_subplot(144)

ax1.plot(ytest1[:ren1], y1, label='measured value')
ax1.plot(predict_test1[:ren1], y1, label='predicted value')
ax2.plot(ytest2[:ren2], y2, label='measured value')
ax2.plot(predict_test2[:ren2], y2, label='predicted value')
ax3.plot(ytest3[:ren3], y3, label='measured value')
ax3.plot(predict_test3[:ren3], y3, label='predicted value')
ax4.plot(ytest4[:ren4], y4, label='measured value')
ax4.plot(predict_test4[:ren4], y4, label='predicted value')
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.title.set_text('ZDNC for test 1')
ax2.title.set_text('ZDNC for test 2')
ax3.title.set_text('ZDNC for test 3')
ax4.title.set_text('ZDNC for test 4')

loc1 = range(150, 10000, 100)
ax1.text(3.1, loc1[0], 'mse  = {:,.3f}'.format(mse1), fontsize=9, alpha=0.9)
ax1.text(3.1, loc1[1], 'rmse = {:,.3f}'.format(rmse1), fontsize=9, alpha=0.9)
ax1.text(3.1, loc1[2], 'mae  = {:,.3f}'.format(mae1), fontsize=9, alpha=0.9)
ax1.text(3.1, loc1[3], 'mape= {:,.3f}'.format(mape1), fontsize=9, alpha=0.9)
ax1.text(3.1, loc1[4], 'r2      = {:,.3f}'.format(r21), fontsize=9, alpha=0.9)

loc2 = range(250, 10000, 200)
ax2.text(2.55, loc2[0], 'mse  = {:,.3f}'.format(mse2), fontsize=9, alpha=0.9)
ax2.text(2.55, loc2[1], 'rmse = {:,.3f}'.format(rmse2), fontsize=9, alpha=0.9)
ax2.text(2.55, loc2[2], 'mae  = {:,.3f}'.format(mae2), fontsize=9, alpha=0.9)
ax2.text(2.55, loc2[3], 'mape= {:,.3f}'.format(mape2), fontsize=9, alpha=0.9)
ax2.text(2.55, loc2[4], 'r2      = {:,.3f}'.format(r22), fontsize=9, alpha=0.9)

loc3 = range(1500, 10000, 100)
ax3.text(1.1, loc3[0], 'mse  = {:,.3f}'.format(mse3), fontsize=9, alpha=0.9)
ax3.text(1.1, loc3[1], 'rmse = {:,.3f}'.format(rmse3), fontsize=9, alpha=0.9)
ax3.text(1.1, loc3[2], 'mae  = {:,.3f}'.format(mae3), fontsize=9, alpha=0.9)
ax3.text(1.1, loc3[3], 'mape= {:,.3f}'.format(mape3), fontsize=9, alpha=0.9)
ax3.text(1.1, loc3[4], 'r2      = {:,.3f}'.format(r23), fontsize=9, alpha=0.9)

loc4 = range(150, 10000, 100)
ax4.text(2.55, loc4[0], 'mse  = {:,.3f}'.format(mse4), fontsize=9, alpha=0.9)
ax4.text(2.55, loc4[1], 'rmse = {:,.3f}'.format(rmse4), fontsize=9, alpha=0.9)
ax4.text(2.55, loc4[2], 'mae  = {:,.3f}'.format(mae4), fontsize=9, alpha=0.9)
ax4.text(2.55, loc4[3], 'mape= {:,.3f}'.format(mape4), fontsize=9, alpha=0.9)
ax4.text(2.55, loc4[4], 'r2      = {:,.3f}'.format(r24), fontsize=9, alpha=0.9)

plt.show()

measured_reshape = np.reshape(ytest1, (len(ytest1), 1))
rows1 = list(np.concatenate([measured_reshape, predict_test1], axis=1).astype('float32'))
measured_reshape = np.reshape(ytest2, (len(ytest2), 1))
rows2 = list(np.concatenate([measured_reshape, predict_test2], axis=1).astype('float32'))
measured_reshape = np.reshape(ytest3, (len(ytest3), 1))
rows3 = list(np.concatenate([measured_reshape, predict_test3], axis=1).astype('float32'))
measured_reshape = np.reshape(ytest4, (len(ytest4), 1))
rows4 = list(np.concatenate([measured_reshape, predict_test4], axis=1).astype('float32'))

np.savetxt("test1_result.csv", rows1, delimiter=",")
np.savetxt("test2_result.csv", rows2, delimiter=",")
np.savetxt("test3_result.csv", rows3, delimiter=",")
np.savetxt("test4_result.csv", rows4, delimiter=",")
