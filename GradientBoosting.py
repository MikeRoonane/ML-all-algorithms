import numpy as np
from main import split_data,print_loss,plot_graph
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

x_train,y_train,_,_,x_test,y_test=split_data(x,y,0.8,0,0.2)

GradientBoostModel=GradientBoostingRegressor()
GradientBoostModel.fit(x_train,y_train)

y_pred=GradientBoostModel.predict(x_train)

y_pred_test=GradientBoostModel.predict(x_test)

print_loss(y_train.squeeze(),y_pred)

plot_graph(x_train,y_train,x_test,y_test,y_pred,y_pred_test)
