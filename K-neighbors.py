import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from main import split_data,plot_graph,print_loss

np.random.seed(42)
x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

x_train,y_train,_,_,x_test,y_test=split_data(x,y,0.8,0,0.2)

model=KNeighborsRegressor(n_neighbors=2)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
y_pred_test=model.predict(x_test)

plot_graph(x_train,y_train,x_test,y_test,y_pred,y_pred_test)
print_loss(y_train,y_pred)