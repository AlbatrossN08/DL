import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import LSTM, Dense,Activation,Dropout

from google.colab import files
uploaded = files.upload()


filename='all_stocks_2006-01-01_to_2018-01-01.csv'
data=pd.read_csv(filename)
data.head()

data=data[['Date','Close','Name']]
multi_ts=data.pivot_table(columns='Name',values='Close',index='Date')
multi_ts.fillna(0,inplace=True)
print(multi_ts.shape)
print(multi_ts.head())

scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(multi_ts)
y_min=min(multi_ts['AABA'])
y_max=max(multi_ts['AABA'])

train_fraction=0.95
train_size=int(len(dataset)*train_fraction)
test_size=len(dataset)-train_size
train,test=dataset[0:train_size,],dataset[train_size:len(dataset),]
print(train.shape)
print(test.shape)

def create_dataset(dataset,window_size=1):
    data_x,data_y=[],[]
    for i in range(len(dataset)-window_size-1):
    a=dataset[i:(i+window_size),]
data_x.append(a)
data_y.append(dataset[i+window_size,0])
return(np.array(data_x),np.array(data_y))
window_size=10
train_x,train_y=create_dataset(train,window_size)
test_x,test_y=create_dataset(test,window_size)
train_x.shape,train_y.shape
pd.DataFrame(train_x[:,:,0]).head()
train_x=np.reshape(train_x,(train_x.shape[0],window_size,train_x.shape[2]))
test_x=np.reshape(test_x,(test_x.shape[0],window_size,test_x.shape[2]))

model=Sequential()
model.add(LSTM(units=50,input_dim=train_x.shape[2],input_length=window_size))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.output_shape
plot_model(model)

model.fit(train_x,train_y,epochs=3,batch_size=1,verbose=1)

testPredict=model.predict(test_x)

def inverse_transform(data,ymax,ymin):
newdata=ymin+data*(ymax-ymin)
return(newdata)

testPredict=inverse_transform(testPredict,y_max,y_min)
test_y=inverse_transform(test_y,y_max,y_min)
testPredict[:,0]

test_y
testScore=math.sqrt(mean_squared_error(test_y,testPredict))
print('Test Score : %.2f RMSE'%(testScore))
