import numpy as np

X= np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])

W1 = np.random.randn(2,2) * 0.01
b1 = np.zeros((1,2))
print(W1)
print(b1)

W2 = np.random.randn(2,1)*0.01
b2=np.zeros((1,1))
print(W2)
print(b2)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def forward_propagation(X,W1,b1,W2,b2):
  Z1=np.dot(X,W1)+b1
  A1=sigmoid(Z1)
  Z2=np.dot(A1,W2)+b2
  A2=sigmoid(Z2)
  return(A2,Z1)

def cost_function(A2,Y):
  m=Y.shape[0]
  cost=-np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))/m
  return cost

def backpropagation(X,Y,A1,A2,W1,W2):
  m=Y.shape[0]
  dz2=A2-Y
  dw2=np.dot(A1.T,dz2)/m
  db2=np.sum(dz2,axis=0,keepdims=True)/m
  dz1=np.dot(dz2,W2.T)*sigmoid_derivative(A1)
  dw1=np.dot(X.T,dz1)/m
  db1=np.sum(dz1,axis=0,keepdims=True)/m
  return dw1,db1,dw2,db2

num_iterations=20000
learning_rate=0.1
for i in range (num_iterations):
  A2,Z1=forward_propagation(X,W1,b1,W2,b2)
  cost=cost_function(A2,Y)
  dw1,db1,dw2,db2=backpropagation(X,Y,sigmoid(Z1),A2,W1,W2)
  W1-=learning_rate*dw1
  b1-=learning_rate*db1
  W2-=learning_rate*dw2
  b2-=learning_rate*db2

X=np.array([[0,1]])
Y=np.array([[1]])
print("Input","Actual Output", "Predictions")
for x,y in zip(X,Y):
  A2,Z1 = forward_propagation(X,W1,b1,W2,b2)
  cost=cost_function(A2,Y)
  dw1,db1,dw2,db2=backpropagation(X,Y,sigmoid(Z1),A2,W1,W2)
  if (np.mean(A2)>0.5):
    predicted=1
  else:
    predicted=0
print(x,y,"    ",predicted)    
