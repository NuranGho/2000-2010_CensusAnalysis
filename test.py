import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


with open('responserate10.csv', 'r') as f:
    responserate10 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\responserate10.csv', skiprows=0, delimiter=',')

x = responserate10['Response Rate10']
#print(x)
    #response_data10 = list(csv.reader(f, delimiter=','))
#print(response_data10[:63])

#y = np.array(pop_data10)

with open('Population2010.csv', 'r') as f:
    Population2010 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\Population2010.csv', skiprows=0, delimiter=',')

y = Population2010['Popluation 2010']
print(y)

x = np.array(x).reshape((-1, 1))
print(x)

y = np.array(y)
print(y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

print("intercept:", model.intercept_)
print("slope:", model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print("new intercept:", new_model.intercept_)
print("new slope:", new_model.coef_)

y_pred = model.predict(x)
print("predicted response:", y_pred, sep='\n')

#new_model.coef_, new_model.intercept_, r_sq = stats.linregress(x,y)
slope = new_model.coef_
intercept = new_model.intercept_
line = slope*x+intercept

x = np.array(x)
y = np.array(y)

plt.plot(x, y, 'o')
plt.plot(x, slope*x + intercept)
plt.title('Linear Regression Analysis: Response Rate & Population 2010')
plt.xlabel('Response Rates', color='#1C2833')
plt.ylabel('Population')
plt.show()

#plt.scatter(x, y, color = "black")
#plt.plot(x, y, y_pred, color = "blue")
#x = np.linspace(-5,5,100)
#y = 2*x+1
#plt.plot(x, y, '-r', label='y=2x+1')
#plt.title('Graph of y=2x+1')
#plt.xlabel('x', color='#1C2833')
#plt.ylabel('y', color='#1C2833')
#plt.legend(loc='upper left')
#plt.grid()
