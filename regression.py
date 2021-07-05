import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib.scale import get_scale_names
from matplotlib.axes import Axes, Subplot

with open('All_data.csv', 'r') as f:
    allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')

x = allData['Population_00']
y = allData['Response_Rate_00']

x = np.array(x).reshape((-1, 1))
print(x)

y = np.array(y)
print(y)

model = LinearRegression().fit(x, y.reshape((-1, 1)))
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

print("intercept:", model.intercept_)
print("slope:", model.coef_)

y_pred = model.predict(x)
print("predicted response:", y_pred, sep='\n')

slope = model.coef_
intercept = model.intercept_
line = slope*x+intercept

model = smf.ols('y ~ x', data=allData).fit()
print(model.summary())
print(model.pvalues)
