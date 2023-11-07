import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow.keras.layers as tfl
import tensorflow.keras.models as tfm
import tensorflow.keras.optimizers as tfo

x = tfl.Input(shape=(784), name="encoder_input")
encoder_dense_layer1 = tfl.Dense(units=300, name="encoder_dense_1")(x)
encoder_activ_layer1 =tfl.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)
encoder_dense_layer2 = tfl.Dense(units=2,
name="encoder_dense_2")(encoder_activ_layer1)
encoder_output = tfl.LeakyReLU(name="encoder_output")(encoder_dense_layer2)
encoder = tfm.Model(x, encoder_output, name="encoder_model")

decoder_input = tfl.Input(shape=(2), name="decoder_input")
decoder_dense_layer1 = tfl.Dense(units=300,
name="decoder_dense_1")(decoder_input)
decoder_activ_layer1 =tfl.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)
decoder_dense_layer2 = tfl.Dense(units=784,
name="decoder_dense_2")(decoder_activ_layer1)
decoder_output = tfl.LeakyReLU(name="decoder_output")(decoder_dense_layer2)
decoder = tfm.Model(decoder_input, decoder_output, name="decoder_model")

ae_input = tfl.Input(shape=(784), name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)
ae = tfm.Model(ae_input, ae_decoder_output, name="AE")

ae.compile(loss="mse", optimizer=tfo.Adam(learning_rate=0.0005))
(x_train_orig, y_train), (x_test_orig, y_test) = mnist.load_data()
x_train_orig = x_train_orig.astype("float32") / 255.0
x_test_orig = x_test_orig.astype("float32") / 255.0
x_train = np.reshape(x_train_orig, newshape=(x_train_orig.shape[0],
np.prod(x_train_orig.shape[1:])))
x_test = np.reshape(x_test_orig, newshape=(x_test_orig.shape[0],
np.prod(x_test_orig.shape[1:])))

ae.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True,
validation_data=(x_test, x_test))
encoded_images = encoder.predict(x_train)
decoded_images = decoder.predict(encoded_images)

decoded_images_orig = np.reshape(decoded_images,
newshape=(decoded_images.shape[0], 28, 28))
num_images_to_show = 3
for im_ind in range(num_images_to_show):
  plot_ind = im_ind*2 + 1
  rand_ind = np.random.randint(low=0, high=x_train.shape[0])
  plt.subplot(num_images_to_show, 2, plot_ind)
  plt.imshow(x_train_orig[rand_ind, :, :], cmap="gray")
  plt.subplot(num_images_to_show, 2, plot_ind+1)
  plt.imshow(decoded_images_orig[rand_ind, :, :], cmap="gray")
