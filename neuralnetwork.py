import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from main import split_data,plot_graph,print_loss
import matplotlib.pyplot as plt

model=nn.Sequential(
    nn.Linear(1,8),
    nn.ReLU(),
    nn.Linear(8,12),
    nn.ReLU(),
    nn.Linear(12,1)

)

np.random.seed(42)
x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

x_train,y_train,_,_,x_test,y_test=split_data(x,y,0.8,0,0.2)

x_train=torch.tensor(x_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)

x_test=torch.tensor(x_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32)

loss_fn=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
n_epochs=100
batch_size=10

for epoch in range(n_epochs):
    for i in range(0,len(x_train),batch_size):
        x_batch=x_train[i:i+batch_size]
        y_pred=model(x_batch)
        y_batch=y_train[i:i+batch_size]
        loss=loss_fn(y_pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss : {loss.item()}")