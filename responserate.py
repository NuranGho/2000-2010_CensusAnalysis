
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import string

response2000 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\2000ResponseRates.csv', skiprows=0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

response2010 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\responserate2010.csv', skiprows=0 )
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

combinedresponse = response2000.merge(response2010, on='NAME', how='inner')
combinedresponse['DIFFERENCE'] = combinedresponse["2010"] - combinedresponse['2000']
print(combinedresponse)

plt.scatter(x=combinedresponse['DIFFERENCE'], y=combinedresponse['NAME'])
plt.xlabel('Difference of Response Rate 2010-2000')
plt.title('Difference of Response Rate in New York from 2010 to 2000.')
plt.ylabel('NAME')
plt.show()
