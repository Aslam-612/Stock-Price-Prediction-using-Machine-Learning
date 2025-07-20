import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
print(df.head())

print(df.info())
print(df.isnull().sum())

df['Target']=df['Close'].shift(-1)
df=df.dropna()

X=df[['Close']]
Y=df['Target']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(Y_test, predictions)
rmse=np.sqrt(mean_squared_error(Y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(Y_test.values, label='Actual')
plt.plot(predictions, label='predicted')
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("price")
plt.show()
