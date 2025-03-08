import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from main import split_data,print_loss,plot_graph
np.random.seed(42)

x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

x_train,y_train,_,_,x_test,y_test=split_data(x,y,0.8,0,0.2)

poly=PolynomialFeatures(degree=5,include_bias=False)
poly_features = poly.fit_transform(x_train.reshape(-1,1))
poly_reg_model=LinearRegression()
poly_reg_model.fit(poly_features,y_train)
y_pred=poly_reg_model.predict(poly_features)

x_test_poly=poly.transform(x_test.reshape(-1,1))

y_test_pred=poly_reg_model.predict(x_test_poly)

print_loss(y_test,y_test_pred)

plot_graph(x_train,y_train,x_test,y_test,y_pred,y_test_pred)