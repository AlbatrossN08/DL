import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
print(X_train.shape)
print(X_test.shape)

noise_factor = 0.2
x_train_noisy = X_train + noise_factor * numpy.random.normal(loc=0.0,
scale=1.0, size=X_train.shape)
x_test_noisy = X_test + noise_factor * numpy.random.normal(loc=0.0,
scale=1.0, size=X_test.shape)
x_train_noisy = numpy.clip(x_train_noisy, 0., 1.)
x_test_noisy = numpy.clip(x_test_noisy, 0., 1.)

model = Sequential()
model.add(Dense(500, input_dim=num_pixels, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(784, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train_noisy, X_train, validation_data=(x_test_noisy, X_test),
epochs=5, batch_size=200)
pred = model.predict(x_test_noisy)
pred.shape

X_test = numpy.reshape(X_test, (10000,28,28)) *255
pred = numpy.reshape(pred, (10000,28,28)) *255
x_test_noisy = numpy.reshape(x_test_noisy, (-1,28,28)) *255
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(1,5,1):
  plt.subplot(2, 10, i+1)
  plt.imshow(X_test[i,:,:], cmap='gray')
  curr_lbl = y_test[i]
  plt.title("(Label: " + str(curr_lbl) + ")")
plt.show()


plt.figure(figsize=(20, 4))
print("Test Images with Noise")
for i in range(1,5,1):
  plt.subplot(2, 10, i+1)
  plt.imshow(x_test_noisy[i,:,:], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Test Images")
for i in range(1,5,1):
  plt.subplot(2, 10, i+1)
  plt.imshow(pred[i,:,:], cmap='gray')
plt.show()
