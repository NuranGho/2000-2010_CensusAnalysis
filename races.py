
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import string

race2000 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\2000RacesUpdated.csv', skiprows=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

race2010 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\2010RacesUpdated.csv', skiprows=1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

combinedRACE = race2000.merge(race2010, on='NAME', how='inner')
combinedRACE['WHITE DIFFERENCE'] = combinedRACE['White_y'] - combinedRACE['White_x']
combinedRACE['Black/African American DIFFERENCE'] = combinedRACE['Black/African American_y'] - combinedRACE['Black/African American_x']
combinedRACE['American Indian & Alaska Native DIFFERENCE'] = combinedRACE['American Indian & Alaska Native_y'] - combinedRACE['American Indian & Alaska Native_x']
combinedRACE['Asian DIFFERENCE'] = combinedRACE['Asian_y'] - combinedRACE['Asian_x']
combinedRACE['Native Hawaiian & Other Pacific Islander DIFFERENCE'] = combinedRACE['Native Hawaiian & Other Pacific Islander_y'] - combinedRACE['Native Hawaiian & Other Pacific Islander_x']
combinedRACE['Hispanic or Latino DIFFERENCE'] = combinedRACE['Hispanic or Latino_y'] - combinedRACE['Hispanic or Latino_x']
print(combinedRACE)

#plt.scatter(x=combinedRACE['WHITE DIFFERENCE'], y=combinedRACE['NAME'])
#plt.xlabel('Difference of Race: White 2010-2000')
#plt.title('Difference of Race: White from 2010 to 2000.')
#plt.ylabel('County')
#plt.show()

#plt.scatter(x=combinedRACE['Black/African American DIFFERENCE'], y=combinedRACE['NAME'])
#plt.xlabel('Difference of Race: Black/African American 2010-2000')
#plt.title('Difference of Race: Black/African American from 2010 to 2000.')
#plt.ylabel('County')
#plt.show()

#plt.scatter(x=combinedRACE['American Indian & Alaska Native DIFFERENCE'], y=combinedRACE['NAME'])
#plt.xlabel('Difference of Race: American Indian & Alaska Native 2010-2000')
#plt.title('Difference of Race: American Indian & Alaska Native from 2010 to 2000.')
#plt.ylabel('County')
#plt.show()

#plt.scatter(x=combinedRACE['Asian DIFFERENCE'], y=combinedRACE['NAME'])
#plt.xlabel('Difference of Race: Asian 2010-2000')
#plt.title('Difference of Race: Asian from 2010 to 2000.')
#plt.ylabel('County')
#plt.show()

#plt.scatter(x=combinedRACE['Native Hawaiian & Other Pacific Islander DIFFERENCE'], y=combinedRACE['NAME'])
#plt.xlabel('Difference of Race: Native Hawaiian & Other Pacific Islander 2010-2000')
#plt.title('Difference of Race: Native Hawaiian & Other Pacific Islander from 2010 to 2000.')
#plt.ylabel('County')
#plt.show()

plt.scatter(x=combinedRACE['Hispanic or Latino DIFFERENCE'], y=combinedRACE['NAME'])
plt.xlabel('Difference of Race: Hispanic or Latino 2010-2000')
plt.title('Difference of Race: Hispanic or Latino from 2010 to 2000.')
plt.ylabel('County')
plt.show()
