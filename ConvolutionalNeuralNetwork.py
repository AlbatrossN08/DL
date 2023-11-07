%tensorflow_version 2.x

import keras
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test)=mnist.load_data()

ind=2025
sample_image=x_train[ind]
pixels=sample_image.reshape((28,28))
plt.imshow(pixels,cmap='gray')
plt.show()

num_classes=10 # 10 classes(0 to 9)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
# Standardizing (0 to 1)
x_train/=255
x_test/=255
# Expand the dimension
x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test, axis=3)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
# Convert class labels into one hot encoded vectors
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
print(x_train.shape)
print(y_train)
print(y_train.shape)

model=Sequential()
model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu',padding='same',name='Hidden_Layer1'))
model.add(MaxPooling2D(pool_size=(2,2),name='Hidden_Layer2'))
model.add(Conv2D(64,(5,5),activation='relu',padding='same',name='Hidden_Layer3'))
model.add(MaxPooling2D(pool_size=(2,2),name='Hidden_Layer4'))
model.add(Flatten(name='Hidden_Layer5'))
model.add(Dense(1024,activation='relu',name='Hidden_Layer6'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu',name='Hidden_Layer7'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax',name='Output_Layer'))

# Compile Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=50,verbose=1)

predicted=model.predict(x_test)

num_images=range(10)
rows=2
columns=5
plt.figsize=(50,50)

for i in (num_images):
  ax=plt.subplot(rows,columns,i+1)
  plt.imshow(x_test[i].reshape((28,28)),cmap='gray')
  title=("Real Label:"+str(np.argmax(y_test[i])))
  label=("Predicted Label:"+str(np.argmax(predicted[i])))
  ax.set_xlabel(label)
  ax.set_title(title)

plt.subplots_adjust(left=0.1,right=2.0,top=1.2,bottom=0.2,wspace=0.5,hspace=0.9)
plt.show()

score=model.evaluate(x_test,y_test,verbose=0)
print('Test Loss : ',score[0])
print('Test Accuracy : ',score[1])
