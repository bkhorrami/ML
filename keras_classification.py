__author__ = 'babak_khorrami'


#-- Feed-Forward Neural Network for Classification 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 15
num_classes = 2
epochs = 20

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = pd.read_csv(url)

id = np.array(range(data.shape[0]))
np.random.shuffle(id)
id_train = id[0:165]
id_test = id[165:]
x  = data.drop(['name','status'], 1)
y = pd.DataFrame(data['status'])

x_train = x.iloc[list(id_train),:].values
y_train = y.iloc[list(id_train),:].values
x_test = x.iloc[list(id_test),:].values
y_test = y.iloc[list(id_test),:].values


y_train = keras.utils.to_categorical(y_train, num_classes =2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

drp_rate = 0.01

model = Sequential()
model.add(Dense(500, activation='relu',input_shape=(22,)))
model.add(Dropout(drp_rate))
model.add(Dense(400, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(400, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(200, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(200, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(200, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(100, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(100, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(20, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(15, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(10, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(10, activation='relu'))
model.add(Dropout(drp_rate))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
yhat=model.predict(x_test,batch_size=15)
print(yhat)
print(y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
