import pandas as pd
import numpy as np
from pandas import plotting
from scipy import stats
from statsmodels.formula.api import ols
import seaborn
from matplotlib import pyplot as plt

## import data
data = pd.read_csv('data/brain_size.csv', header=0, delim_whitespace=True)
data

## numpy arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

## manipulating data
data.shape
data.columns

print(data['Gender'])

data[data['Gender'] == 'Female']['VIQ'].mean()

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender,value.mean()))

groupby_gender.mean()

## plotting data
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])   

plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

## hypothesis testing
#### t test - 1 sample
stats.ttest_1samp(data['VIQ'], 0)

#### t test - 2 samples
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender']== 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

#### paired tests
stats.ttest_ind(data['FSIQ'], data['PIQ'])   

stats.ttest_rel(data['FSIQ'], data['PIQ'])   

stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)   

stats.wilcoxon(data['FSIQ'], data['PIQ'])   


## linear models 
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise 
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# create a dataframe containing all the relevant variables
data = pd.DataFrame({'x': x, 'y': y})
# ols model
model = ols("y ~ x", data).fit()
print(model.summary())  

#### categorial variables - comparison
data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

model = ols('VIQ ~ C(Gender)', data).fit()
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long) 
model = ols("iq ~ type", data_long).fit()
print(model.summary()) 

stats.ttest_ind(data['FSIQ'], data['PIQ'])   

## multiple regression
data2 = pd.read_csv('data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data2).fit()
print(model.summary())

print(model.f_test([0, 1, -1, 0]))  

## seaborn - pairplot
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],kind='reg')

seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],kind='reg', hue='SEX') 
#### matplotlib
plt.rcdefaults()
#### implot
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)

## testing for interactions
result = data_2.ols(formula='wage ~ education + gender + education * gender',data=data).fit()
print(result.summary())