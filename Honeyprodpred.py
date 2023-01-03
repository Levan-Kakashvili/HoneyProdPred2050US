
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year['year']
X = X.values.reshape(-1,1) # we need to reshape data to be able to feed this to sklearn
y = prod_per_year['totalprod']
plt.scatter(X,y,alpha = 0.4)


regr = linear_model.LinearRegression()
regr.fit(X,y)

#y_predict = regr.predict(X) #This was used to find and plot line to see how it predicts with historic data
#plt.plot(X,y_predict)
#plt.show()
X_future = np.array(range(2013,2051)) #adjust years there
X_future = X_future.reshape(-1,1)
future_predict = regr.predict(X_future)
plt.scatter(X_future,future_predict,alpha = 0.4)
plt.title("Honey Production Prediction in US")
plt.xlabel("Years")
plt.ylabel("Production")
plt.show()
