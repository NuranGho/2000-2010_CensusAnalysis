import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df2020 = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\responserate20.json')
df2020 = df2020.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\responserate20.csv', index = None)
df2020 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\responserate20.csv', skiprows=1 )
df2020 = df2020.rename(columns={'CRRALL': 'Response Rates 2020','state': 'STATE', 'county': 'COUNTY', 'tract': 'TRACT'})
formatted2020 = df2020.drop(['CRRINT', 'DRRINT','DRRALL','RESP_DATE',], axis=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print(df1)

df2010 = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\responserate10.json')
df2010 = df2010.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\responserate10.csv', index = None)
df2010 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\responserate10.csv', skiprows=1 )
df2010 = df2010.rename(columns={'FSRRR2010': 'Response Rates 2010','state': 'STATE', 'county': 'COUNTY', 'tract': 'TRACT'})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print(df2)

combinedDF = formatted2020.merge(df2010, on='TRACT', how='inner')
combinedDF['DIFFERENCE'] = combinedDF['FSRR2010'] - combinedDF['Response Rates 2020']


plt.scatter(x=combinedDF['NAME_x'], y=combinedDF['DIFFERENCE'])
plt.show()
