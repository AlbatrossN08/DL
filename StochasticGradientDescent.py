import numpy as np
%matplotlib inline
from sklearn.linear_model import LinearRegression
import pandas as pd

x1=np.random.randint(1,30,200000)
x2=np.random.randint(1,30,200000)
y=4+2*x1+3*x2+3*np.random.random(200000)

w=np.random.random(3)
x=pd.DataFrame({'const':np.ones(200000),'x1':x1,'x2':x2})

lr=LinearRegression()
lr.fit(x.iloc[:,1:],y)

def mypred(features,weights):
  preds=np.dot(features,weights)
  return(preds)

def myerror(target,features,weights):
  preds=mypred(features,weights)
  errors=target-preds
  return(errors)

def mycost(target,features,weights):
  errors=myerror(target,features,weights)
  cost=np.dot(errors.T,errors)
  return(cost)

def gradient(target,features,weights):
  errors=myerror(target,features,weights)
  grad=-np.dot(features.T,errors)/features.shape[0]
  return(grad)
def my_lr_sgd(target,features,learning_rate,num_steps):
  cost=[]
  weights=np.random.random(features.shape[1])
  for i in np.arange(num_steps):
    rand_ind=np.random.choice(range(features.shape[0]),10)
    target_sub=target[rand_ind]
    features_sub=features.iloc[rand_ind,:]
    weights -= learning_rate*gradient(target_sub,features_sub,weights)
    cost.append(mycost(target,features,weights))
  return(cost,weights)

cost_sgd,w_sgd=my_lr_sgd(y,x,.001,1000)
np.log(pd.DataFrame({'cost_sgd':cost_sgd})).plot()
