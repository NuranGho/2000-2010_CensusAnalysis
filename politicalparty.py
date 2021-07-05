
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import string
#2000 dataset for political party based on counties in new york
df2000 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\Political2000.csv', skiprows=0)
#format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#rename the name of the cols
df2000 = df2000.rename(columns={'county': 'COUNTY'})
print(df2000)

df2010 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\Political2010.csv', skiprows=0 )

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

combinedDF = df2000.merge(df2010, on='COUNTY', how='inner')
combinedDF['DIFFERENCE'] = combinedDF['TOTAL'] - combinedDF['Total']



plt.scatter(x=combinedDF['DIFFERENCE'], y=combinedDF['COUNTY'])
plt.xlabel('Difference of Total People Registered to a Party 2010-2000')
plt.title('Difference of Total People Registered to a Party in New York from 2010 to 2000.')
plt.ylabel('County')
plt.show()
