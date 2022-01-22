# -*- coding: utf-8 -*-
"""
Author: Gustavo Venturi
Date: 2021-01-21
Title: Beijing PM2.5 Data Dataset
Description: Estimative the PM2.5 particles behavior based on its answer to other 
parameters (how other parameters affect the PM2.5 particles) 
"""

#!pip install pandas
#!pip install BeautifulSoup4
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

#webscrapping from UCI page

url = 'https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data'

html = urlopen(url)
bs = BeautifulSoup(html,'html.parser')
dwloadUrl = bs.find_all('a', href=True)

dwloadUrl

dl = []
for link in dwloadUrl:
    u = str(link) 
    if u.find('Data Folder') > 0:
        dl.append(link['href'])

dl = str(dl[0]).replace('..', 'https://archive.ics.uci.edu/ml') #found url to direct to repository

html = urlopen(dl)
bs = BeautifulSoup(html,'html.parser')
dwloadUrl = bs.find_all('a', href=True)

dlFile = []
i = 0
for link in dwloadUrl:
    u = str(link)
    #x = u.find('.csv')    
    if u.find('.csv') > 0:
        dlFile.append(link['href'])

dl = dl + str(dlFile[0]) #url download dataset

#get data from webscrapping

df = pd.read_csv(dl)
df

'''
The polluant **PM 2.5** which is particulate matter suspended in air with a size smaller than or equal to 2.5 μm.  
High concentrations of PM 2.5 can cause severe damage to the respiratory system.
According to the WHO (World Health Organization), the tolerable limits of these particles are: 
» Annual average: 10 μg/m³  
» Average 24 hours: 25 μg/m³
Source: https://apps.who.int/iris/bitstream/handle/10665/69477/WHO_SDE_PHE_OEH_06.02_eng.pdf;jsessionid=4BA80EAA77F8DD1CC0C6798EADBF1C8F?sequence=1
'''


## pm2.5 seasonality
#!pip install matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.plot(df['pm2.5'],'-', )
plt.xlabel('registers')
plt.ylabel('pm2.5')
plt.axhline(y=10,color='r',linestyle='dotted')
plt.axhline(y=25,color='r',linestyle='-')
plt.title('Evolution by entry of pm2.5')
plt.show();


from scipy.stats import norm

maximus = pd.Series(df['pm2.5'],name='pm2.5').max()
minimus = pd.Series(df['pm2.5'],name='pm2.5').min()

x_axis = np.arange(minimus, maximus, 10)

mean = pd.Series(df['pm2.5'],name='pm2.5').mean()
sd = pd.Series(df['pm2.5'],name='pm2.5').std()

plt.plot(x_axis, norm.pdf(x_axis,mean,sd))
plt.title('Normal Distribution pm2.5')
plt.axvline(x=mean,color='r',linestyle='-')
plt.axvline(x=mean+sd,color='r',linestyle='-.')
plt.axvline(x=mean+(sd*2),color='r',linestyle='dotted')
plt.axvline(x=mean-sd,color='r',linestyle='-.')
plt.axvline(x=mean-(sd*2),color='r',linestyle='dotted')
plt.show()


## monthly analysis

df.describe()

dfYear = df[['year','month','pm2.5']]
dfYear['Period'] = (dfYear['year'])*100+(dfYear['month'])
dfYear = dfYear.drop(columns=['year','month'])
dfYearM = dfYear.groupby(by=['Period']).mean()
dfYearM

#monthly measures chart
height = dfYearM['pm2.5']
x_pos = np.arange(60)
plt.bar(x_pos, height)
plt.xlabel('months')
plt.ylabel('pm2.5')
plt.title('Monthly average pm2.5')
plt.axhline(y=10,color='r',linestyle='dotted')
plt.axhline(y=25,color='r',linestyle='-')
plt.show()

## correlation between features

#!pip install seaborn
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            linewidths=1,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


dfHour = df[['hour','month','pm2.5']]
dfHour = dfHour.groupby(by=['hour','month']).mean()
dfHour

height = dfHour['pm2.5']
x_pos = np.arange(288)
plt.bar(x_pos, height)
plt.xlabel('hours')
plt.ylabel('pm2.5')
plt.title('Hourly average per month pm2.5')
plt.axhline(y=10,color='r',linestyle='dotted')
plt.axhline(y=25,color='r',linestyle='-')
plt.show()

#features more influents with pm2.5
#Dew Point correlation 0.17423
#Iws (Cumulated wind speed) correlation -0.247784

dfMostInf = df[['pm2.5','DEWP','Iws']]

plt.scatter(dfMostInf['pm2.5'], dfMostInf['DEWP'] )
plt.xlabel('pm2.5')
plt.ylabel('Dew Point (ºC)')
plt.title('Comparisson between pm2.5 and Dew Pont')
plt.axvline(x=10,color='r',linestyle='dotted')
plt.axvline(x=25,color='r',linestyle='-')
plt.show()

'''
Conclusions:
Airborne particles can be captured as condensation nuclei as dew condenses, whereas gases or liquid particles might dissolve into the dewdrops. Thus, dew formation can actually help purify urban air, and dew is recognized as the sink of nighttime moisture and near-surface particulate matter (e.g., PM2.5 and PM10).
Source: https://www.hindawi.com/journals/amete/2017/3514743/
'''

plt.scatter(dfMostInf['pm2.5'], dfMostInf['Iws'] )
plt.xlabel('pm2.5')
plt.ylabel('Cumulated Wind Speed (m/s)')
plt.title('Comparisson between pm2.5 and Cumulated Wind Speed')
plt.axvline(x=10,color='r',linestyle='dotted')
plt.axvline(x=25,color='r',linestyle='-')
plt.show()

'''
Conclusions:
More wind register less particules suspended in air.
'''

sns.pairplot(df)
'''
Conclusions:
Higher cumulated rain and snow register less particules in air.
'''

# predict pm2.5
#!pip install sklearn

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#split data and target
df['pm2.5'] = df['pm2.5'].fillna(mean)
X = df.drop(columns=['pm2.5','No'])
X['cbwd'].value_counts()
X = pd.get_dummies(X,columns=['cbwd'],dummy_na=True)

y = df['pm2.5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

params = {
    "n_estimators": 1000,
    "max_depth": 10,
    "min_samples_split": 8,
    "learning_rate": 0.01,
    "loss": "absolute_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_percentage_error
mae = mean_absolute_percentage_error(y_test, reg.predict(X_test))
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))

#plot training deviance
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

#feature importance
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

#estimate pm2.5 for an entry:

#
# FILL VALUES
#

YEAR = 2021
MONTH = 1
DAY = 22
HOUR = 12
DEWP = -5.0
TEMP = 15.0
PRESS = 996.5
IWS = 350.5
IS = 0
IR = 155
CBWD_NE = 1
CBWD_NW = 0
CBWD_SE = 0
CBWD_CV = 0
CBWD_NAN = 0

app = {'year': {1: YEAR},
 'month': {1: MONTH},
 'day': {1: DAY},
 'hour': {1: HOUR},
 'DEWP': {1: DEWP},
 'TEMP': {1: TEMP},
 'PRES': {1: PRESS},
 'Iws': {1: IWS},
 'Is': {1: IS},
 'Ir': {1: IR},
 'cbwd_NE': {1: CBWD_NE},
 'cbwd_NW': {1: CBWD_NW},
 'cbwd_SE': {1: CBWD_SE},
 'cbwd_cv': {1: CBWD_CV},
 'cbwd_nan': {1: CBWD_NAN}}

app2 = pd.DataFrame.from_dict(app)


reg.predict(app2)
