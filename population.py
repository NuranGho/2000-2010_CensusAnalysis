
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import string

pop2000 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\2000Population.csv', skiprows=0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

pop2010 = pd.read_csv(r'C:\Users\nhoss\OneDrive\Desktop\Senior_Project\2010Population.csv', skiprows=0 )
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

combinedPOP = pop2000.merge(pop2010, on='NAME', how='inner')
combinedPOP['DIFFERENCE'] = combinedPOP['2010'] - combinedPOP['2000']
print(combinedPOP)

plt.scatter(x=combinedPOP['DIFFERENCE'], y=combinedPOP['NAME'])
plt.xlabel('Difference of Population 2010-2000')
plt.title('Difference of Population in New York from 2010 to 2000.')
plt.ylabel('County')
plt.show()
