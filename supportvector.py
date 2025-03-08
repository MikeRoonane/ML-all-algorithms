import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from main import split_data,print_loss,plot_graph
from sklearn.svm import SVR

sc_X=StandardScaler()
sc_Y=StandardScaler()
np.random.seed(42)
x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

x_train,y_train,_,_,x_test,y_test=split_data(x,y,0.8,0,0.2)

x_train_sc=sc_X.fit_transform(x_train)
y_train_sc=sc_Y.fit_transform(y_train.reshape(-1,1))

regressor = SVR(kernel = 'rbf',C=100,gamma=1)
regressor.fit(x_train_sc,y_train_sc.ravel())

y_pred=regressor.predict(x_train_sc)
y_pred=sc_Y.inverse_transform(y_pred.reshape(-1,1))

x_test_sc=sc_X.transform(x_test)
y_pred_test=regressor.predict(x_test_sc)
y_pred_test=sc_Y.inverse_transform(y_pred_test.reshape(-1,1))

print_loss(y_train,y_pred.squeeze())

plot_graph(x_train,y_train,x_test,y_test,y_pred,y_pred_test)