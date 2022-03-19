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

# Multivariable regression using all datasets
X_train, X_test, y_train, y_test = train_test_split(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']], df['PRICE'], test_size=0.2, train_size=0.8, random_state=10)
regression = LinearRegression()
regression.fit(X_train, y_train)
print(f"Accuracy of Regression on trained dataset {regression.score(X_test, y_test)*100}")

# Evaluating regression by observing deviation from predicted relationship
predicted_values = regression.predict(X_train)
residuals = (y_train - predicted_values)
plt.scatter(y=predicted_values, x=y_train)
yt_plot = plt.plot(y_train, y_train, color='r')
plt.ylabel('Actual Prices')
plt.xlabel('Predicted Prices')
plt.show()
res_plot = plt.scatter(y_train, residuals, color='g')
plt.ylabel('Residuals')
plt.xlabel('Predicted Prices')
plt.show()

# Observe skewness of residuals
sns.displot(residuals, kind='kde')
plt.show()


# Investigation to determine if accuracy would increase if the price value used to train the dataset should rather be
# a logarithm of the original dataset

# Comparison of bell curves to determine which would have less skewness and therefore be a better fit for
# linear regression methods used
sns.displot(df.PRICE, kind='kde')
plt.show()
sns.displot(np.log(df.PRICE), kind='kde')
plt.show()

# Redo Regression with log price values
X_train, X_test, y_train, y_test = train_test_split(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']], np.log(df['PRICE']), test_size=0.2, train_size=0.8, random_state=10)
regression2 = LinearRegression()
regression2.fit(X_train, y_train)
print(f"Accuracy of new Regression on trained dataset {regression2.score(X_test, y_test)*100}")

# Test program using random row rom DF
sample = df.sample()
del sample['PRICE']
price_est = np.exp(regression2.predict(sample)[0]) * 1000
print(f'Sample: {sample}\nPrice estimate: {price_est}')