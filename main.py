import random
from math import floor
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
#Randomizing and splitting data into train,val and test
def split_data(x,y,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1):
    assert train_ratio+val_ratio+test_ratio == 1,"Ratio must be 1"

    data=list(zip(x,y))
    random.shuffle(data)

    train_end=floor(len(x)*train_ratio)
    val_end=train_end+floor(len(x)*val_ratio)
    
    X,Y=zip(*data)

    x_train= np.array(X[:train_end])
    y_train= np.array(Y[:train_end])
    x_val=np.array(X[train_end:val_end])
    y_val=np.array(Y[train_end:val_end])
    x_test=np.array(X[val_end:])
    y_test=np.array(Y[val_end:])

    return x_train,y_train,x_val,y_val,x_test,y_test

#Printing losses
def print_loss(y, y_pred):

    y = torch.tensor(y, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    loss_mse = mse_loss_fn(y, y_pred)
    loss_mae = mae_loss_fn(y, y_pred)

    print(f"Mean Squared Error: {loss_mse.item()}, Mean Absolute Error: {loss_mae.item()}")

#Creating graphs

def plot_graph(x_train,y_train,x_test,y_test,y_pred,y_pred_test):
    fig,axis=plt.subplots(1,2,figsize=(12,5))

    axis[0].scatter(x_train,y_train,label="Actual",color="blue")
    axis[0].scatter(x_train,y_pred,label="Predicted",color="red")
    axis[0].set_title("Training dataset")

    axis[1].scatter(x_test,y_test,label="Actual",color="blue")
    axis[1].scatter(x_test,y_pred_test,label="Predicted",color="red")
    axis[1].set_title("Test dataset")

    plt.tight_layout()
    plt.show()