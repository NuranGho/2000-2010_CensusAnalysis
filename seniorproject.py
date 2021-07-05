import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import plotly.graph_objects as go

df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\2010.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\2010.csv', index = None)

df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\RR10.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\RR10.csv', index = None)

#json to csv
#df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\2000RacesUpdated.json')
#df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\2000RacesUpdated.csv', index = None)

#json to csv
df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\2010RacesUpdated.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\2010RacesUpdated.csv', index = None)

#json to csv
df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\2000Population.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\2000Population.csv', index = None)

#json to csv
df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\responserate20.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\responserate20.csv', index = None)

# csv to pandas DF
census2020 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\responserate20.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print(census2020)

df1 = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\responserate10.json')
df1.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\responserate10.csv', index = None)

# csv to pandas DF
census2010 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\responserate10.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print all data
#print(census2010)

plt.scatter(x=combinedDF['NAME_x'], y=combinedDF['DIFFERENCE'])
plt.show()

df = pd.read_json (r'C:\Users\nuran\Desktop\Senior_Project\2000.json')
df.to_csv (r'C:\Users\nuran\Desktop\Senior_Project\2000.csv', index = None)

# csv to pandas DF
census2000 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\2000.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print(census2000)



# csv to pandas DF
census2010 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\2010.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#print(census2010)

#combinedDF = formatted2020.merge(df2010, on='TRACT', how='inner')
#combinedDF['DIFFERENCE'] = combinedDF['FSRR2010'] - combinedDF['Response Rates 2020']

#print(my_data)
