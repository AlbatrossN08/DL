import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
print(data)

def splitSequence(seq, n_steps):
  X = []
  y = []
  for i in range(len(seq)):
    lastIndex = i + n_steps
    if lastIndex > len(seq) - 1:
      break
    seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
    X.append(seq_X)
    y.append(seq_y)
    pass
  X = np.array(X)
  y = np.array(y)
  return X,y
  pass

n_steps = 5
X, y = splitSequence(data, n_steps = 5)
print(X)
print(y)

for i in range(len(X)):
  print(X[i], y[i])

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X[:2])

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps,
n_features)))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=1)

test_data = np.array([27, 29, 31, 33, 35])
test_data = test_data.reshape((1, n_steps, n_features))
test_data
predictNextNumber = model.predict(test_data, verbose=1)
print(np.round(predictNextNumber))
