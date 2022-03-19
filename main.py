import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:,.2f}'.format

df = pd.read_csv('boston.csv')

#Perform Density Estimator to determine the skewness of the datasets
price_kde = sns.displot(data=df, kind='kde', aspect=2, x='PRICE')
price_kde.set(ylabel='Density', xlabel='Price')
rm_kde = sns.displot(data=df, kind='kde', aspect=2, x='RM', color='r')
rm_kde.set(ylabel='Density', xlabel='Average Number of Rooms')
dis_kde = sns.displot(data=df, kind='kde', aspect=2, x='DIS', color='g')
dis_kde.set(ylabel='Density', xlabel='Weighted Distances to five Boston Employment Centres')
rad_kde = sns.displot(data=df, kind='kde', aspect=2, x='RAD', color='y')
rad_kde.set(ylabel='Density', xlabel='Accessibility to Radial Highways')
plt.show()

# Create bar graph of CHAS to see the number of people situated near the Charles river
chas_c = df.groupby('CHAS').count()
chas_c.index = ['No', 'Yes']
chas_bar=px.bar(chas_c, color=chas_c.index)
chas_bar.show()

# Create pair plots to see relationships of information relative to one another to find relationships
sns.pairplot(df[['NOX', 'DIS', 'RM', 'PRICE', 'LSTAT']])

# Use jointplots to plot a linear line to fit to determine if the relationship between datasets are linear
sns.jointplot(data=df, x='LSTAT', y='RM', kind='reg')
sns.jointplot(data=df, x='LSTAT', y='PRICE', kind='reg')
plt.show()
# print(f"NaN Values: {df.isna().values.any()}\nDuplicates: {df.duplicated.values.any()}")
