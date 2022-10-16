#!/usr/bin/env python
# coding: utf-8

# ### California Housing Prices Data Modeling
# 
# * [importing the libraries](#importing-the-libraries)
# * [Reading the data](#Reading-the-data)
# * [Exploring the data](#Exploring-the-data)
# * [Visualizing the data](#Visualizing-the-data)
# * [Drop duplicates values](#Drop-duplicates-values)
# * [Missing data](#Missing-data)
# * [Filling in missing values](#Filling-in-missing-values)
# * [Outlier and Deleting Observations](#Outlier-and-Deleting-Observations)
# * [Encoding categorical features](#Encoding-categorical-features)
# * [Drop unimportant columns](#Drop-unimportant-columns)
# * [Scaling and Split the data](#Scaling-and-Split-the-data)
# * [Linear Regression](#Linear-Regression)
# * [Ridge regression](#Ridg-regression)
# * [Lasso Regression](#Lasso-Regression)
# * [compersion between models](#compersion-between-models)

# ![image.png](attachment:image.png)

# ## importing the libraries

# In[145]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ## Reading the data

# In[146]:


df = pd.read_csv(r"D://Samsung Innovation Campus//Data//housing price//housing.csv")


# ## Exploring the data

# In[147]:


pd.set_option("display.max.columns", None)
df.sample(5)


# In[148]:


df.info()


# In[149]:


df.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'})


# In[150]:


((df.describe(include='object')).T).style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'})


# ## Visualizing the data

# In[151]:


plt.figure(figsize = (8,8))
plt.scatter(df['longitude'] , df['latitude'] , c = df['median_house_value'])
plt.colorbar()
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("House Prices")


# In[152]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap='viridis')

plt.show()


# > **Target variable median_house_value is very mildly correlated to all but one feature here: median_income, so one might outline this as an important feature.**

# In[153]:


df.hist(bins = 50 , figsize=(30 , 20),color="k")
plt.show()


# > ****plot histogram to observe the data Distribution****  

# In[154]:


num_columns = list(df.select_dtypes(include=["int64","float64"]).columns)[2:]

fig, ax = plt.subplots(4,2, figsize = (15,15))
font_dict = {'fontsize': 14}
ax = np.ravel(ax)

for i in num_columns:
    sns.boxplot(data=df,x=i,ax=ax[num_columns.index(i)],palette="plasma").set_title(i)

ax = np.reshape(ax, (4, 2))
plt.tight_layout()
plt.show()


# > ****plot boxplot to observe the data Distribution and outliers****

# ## Drop duplicates values

# In[155]:


print("Data shape Before duplicates Values: ",df.shape)
df_new=df.drop_duplicates()
print("Data shape After duplicates Values: ",df_new.shape)


# > **We note that there are no duplicates Values**

# ## Missing data

# ### See how many missing data points we have

# In[156]:


missing_values_df = df.isnull().sum()
print(missing_values_df)


# > **total_bedrooms is the only Column to have missing values in our dataset**

# In[157]:


#Percentage of missing data by feature

df_na = (missing_values_df / len(df)) * 100

# drop columns without missing values 
df_na = df_na.drop(df_na[df_na == 0].index)

#sort
df_na=df_na.sort_values(ascending=False)

print("Percentage of missing values in {} : {} %".format(df_na.index[0],df_na[0]))

total_cells = np.product(df.shape)

total_missing = missing_values_df.sum()

print("percent of data that is missing from all Dataset: {}%".format(round((total_missing/total_cells) * 100,ndigits=2)))


# In[158]:


# create plot
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.isnull())
ax.set(title='heatmap of missing data by feature', xlabel='feature')
plt.show()


# ## Filling in missing values

# In[159]:


df.sample(5)


# In[160]:


def Zscore_outlier(column,df,scale=3.75):
    out=[]
    m = np.mean(df[column])
    sd = np.std(df[column])
    for i in df[column]: 
        z = (i-m)/sd
        if np.abs(z) > scale: 
            out.append(i)
    df=df[df[column].isin(out) == False]
    return df


# In[161]:


fig, ax = plt.subplots(1,2, figsize = (12,4))
font_dict = {'fontsize': 14}
ax = np.ravel(ax)

bedrooms_total=pd.DataFrame(df_new['total_bedrooms']/df_new['total_rooms'],columns=["bedrooms_total"])

print("bedrooms to total ratio mean : ",bedrooms_total.mean()[0])

sns.boxplot(data=bedrooms_total,x='bedrooms_total',palette='viridis', ax=ax[0]).set_title('bedrooms to total ratio before Outlier')

bedrooms_total=Zscore_outlier(column='bedrooms_total',df=bedrooms_total)

print("bedrooms to total ratio mean : ",bedrooms_total.mean()[0])

sns.boxplot(data=bedrooms_total,x='bedrooms_total',palette='cividis', ax=ax[1]).set_title('bedrooms to total ratio after drop Outlier')


ax = np.reshape(ax, (1, 2))
plt.tight_layout()
plt.show()
bedrooms_total=bedrooms_total.mean()[0]


# In[162]:


df_new['total_bedrooms']=df_new['total_bedrooms'].fillna(bedrooms_total*df_new['total_rooms'])


# In[163]:


df_new.isnull().sum()


# ### Plot the Distribution of total_bedrooms before and after Filling in missing values

# In[164]:


fig, ax = plt.subplots(2,2, figsize = (12,8))
font_dict = {'fontsize': 14}

ax = np.ravel(ax)
sns.kdeplot(data=df_new,x='total_bedrooms',ax = ax[0]).set_title('After Distribution')
sns.kdeplot(data=df,x='total_bedrooms',ax = ax[1]).set_title('Before Distribution')
sns.boxplot(data=df_new,x='total_bedrooms', ax=ax[2],palette='viridis').set_title('After Distribution')
sns.boxplot(data=df,x='total_bedrooms', ax=ax[3],palette='cividis').set_title('Before Distribution')

ax = np.reshape(ax, (2, 2))
plt.tight_layout()
plt.show()


# ## Outlier and Deleting  Observations

# In[165]:


data=df_new.copy()
df_new.shape


# In[166]:


def Distribution2(columne,data,i):
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    font_dict = {'fontsize': 14}
    title=['Before Distribution','After Distribution']
    ax = np.ravel(ax)
    if i==1:
        sns.set(style='whitegrid')
        sns.kdeplot(data=data,x=columne ,ax = ax[0],color='r').set_title(title[i])
        sns.boxplot(data=data,x=columne ,ax = ax[1],palette='magma').set_title(title[i])
        sns.scatterplot(data=data,x=columne ,ax = ax[2], y=data['median_house_value'],color='r').set_title(title[i])
    else:
        sns.set(style='whitegrid')
        sns.kdeplot(data=data,x=columne ,ax = ax[0],color='#2171b5').set_title(title[i])
        sns.boxplot(data=data,x=columne ,ax = ax[1],color='#2171b5').set_title(title[i])
        sns.scatterplot(data=data,x=columne ,ax = ax[2], y=data['median_house_value'],color='#2171b5').set_title(title[i])
    
    ax = np.reshape(ax, (1, 3))
    plt.tight_layout()


# ### 1. total_bedrooms

# In[167]:


Distribution2(columne='total_bedrooms',data=data,i=0)


# In[168]:


data[data['total_bedrooms']>=2800].shape


# In[169]:


data=data[data['total_bedrooms']<2800]
Distribution2(columne='total_bedrooms',data=data,i=1)


# ### 2. total_rooms

# In[170]:


Distribution2(columne='total_rooms',data=data,i=0)


# In[171]:


data[data['total_rooms']>=15000].shape


# In[172]:


data=data[data['total_rooms']<15000]
Distribution2(columne='total_rooms',data=data,i=1)


# ### 3. housing_median_age

# In[173]:


Distribution2(columne='housing_median_age',data=data,i=0)


# > **no Outliers**

# ### 4. population

# In[174]:


Distribution2(columne='population',data=data,i=0)


# In[175]:


data[data['population']>=6000].shape


# In[176]:


data=data[data['population']< 6000]
Distribution2(columne='population',data=data,i=1)


# ### 5. households

# In[177]:


Distribution2(columne='households',data=data,i=0)


# In[178]:


data[data['households']>=2000].shape


# In[179]:


data=data[data['households']<2000]
Distribution2(columne='households',data=data,i=1)


# ### 6. median_income

# In[180]:


Distribution2(columne='median_income',data=data,i=0)


# In[181]:


data[data['median_income']>=9].shape


# In[182]:


data=data[data['median_income']< 9]
Distribution2(columne='median_income',data=data,i=1)


# In[183]:


data.shape


# ## About Categorical Data
# 
# ### What is Categorical Data?
# Categorical data are variables that contain label values rather than numeric values.
# 
# The number of possible values is often limited to a fixed set.
# 
# Categorical variables are often called nominal.
# 
# Some examples include:
# 
# A “pet” variable with the values: “dog” and “cat“. A “color” variable with the values: “red“, “green” and “blue“. A “place” variable with the values: “first”, “second” and “third“. Each value represents a different category.
# 
# Some categories may have a natural relationship to each other, such as a natural ordering.
# 
# The “place” variable above does have a natural ordering of values. This type of categorical variable is called an ordinal variable.
# 
# ### What is the Problem with Categorical Data?
# Some algorithms can work with categorical data directly.
# 
# For example, a decision tree can be learned directly from categorical data with no data transform required (this depends on the specific implementation).
# 
# Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.
# 
# In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves.
# 
# This means that categorical data must be converted to a numerical form. If the categorical variable is an output variable, you may also want to convert predictions by the model back into a categorical form in order to present them or use them in some application.
# 
# ## Treatment Techniques for Categorical Data
# 
# ### How to Convert Categorical Data to Numerical Data?
# This involves two steps:
# 
# #### 1. Integer Encoding
# As a first step, each unique category value is assigned an integer value.
# 
# For example, “red” is 1, “green” is 2, and “blue” is 3.
# 
# This is called a label encoding or an integer encoding and is easily reversible.
# 
# For some variables, this may be enough.
# 
# The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.
# 
# For example, ordinal variables like the “place” example above would be a good example where a label encoding would be sufficient.
# 
# #### 2. One-Hot Encoding
# For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.
# 
# In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
# 
# In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.
# 
# In the “color” variable example, there are 3 categories and therefore 3 binary variables are needed. A “1” value is placed in the binary variable for the color and “0” values for the other colors.
# 
# For further reference : https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

# ## Encoding categorical features

# ### One hot encoding implementation
# ![Introduction-to-Python_Watermarked.48eeee4e1109.jpg](attachment:Introduction-to-Python_Watermarked.48eeee4e1109.jpg)

# In[184]:


one_hot_encoders_data =  pd.get_dummies(data)
one_hot_encoders_data.head()


# In[185]:


# a structured approach
cols = one_hot_encoders_data.columns
data = pd.DataFrame(one_hot_encoders_data,columns= cols)
data.sample(5)


# ## Drop unimportant columns

# In[186]:


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap='viridis')

plt.show()


# In[187]:


data.drop(['longitude','latitude'],axis=1,inplace=True)


# In[188]:


data.info()


# ##  Scaling and Split the data

# In[189]:


data.columns


# In[190]:


data_=data[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
       'households', 'median_income','ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN', 'median_house_value']]


# In[191]:


x = data_.drop(['median_house_value'] , axis = 1)
y= data_['median_house_value' ].values


# ### Scaling
# - **StandardScaler** follows Standard Normal Distribution (SND). Therefore, it makes mean = 0 and scales the data to unit variance. 
# - **MinMaxScaler** scales all the data features in the range [0, 1] or else in the range [-1, 1] if there are negative values in the dataset. This scaling compresses all the inliers in the narrow range [0, 0.005].
# 
# In the presence of outliers, StandardScaler does not guarantee balanced feature scales, due to the influence of the outliers while computing the empirical mean and standard deviation. This leads to the shrinkage in the range of the feature values. 
# - By using **RobustScaler()**, we can remove the outliers and then use either StandardScaler or MinMaxScaler for preprocessing the dataset. 

# In[193]:


from sklearn import preprocessing
scaler_Robust = preprocessing.RobustScaler()
robust_df = scaler_Robust.fit_transform(x)
robust_df = pd.DataFrame(robust_df,columns=list(x.columns ))


# In[194]:


scaler_MinMax = preprocessing.MinMaxScaler()
x = scaler_MinMax.fit_transform(robust_df)


# # Model Training

# ## What is Regression?

# **Regression is a statistical technique that helps in qualifying the relationship between the interrelated economic variables. The first step involves estimating the coefficient of the independent variable and then measuring the reliability of the estimated coefficient. This requires formulating a hypothesis, and based on the hypothesis, we can create a function.**
# 
# If a manager wants to determine the relationship between the firm’s advertisement expenditures and its sales revenue, he will undergo the test of hypothesis. Assuming that higher advertising expenditures lead to higher sale for a firm. The manager collects data on advertising expenditure and on sales revenue in a specific period of time. This hypothesis can be translated into the mathematical function, where it leads to −
# 
# #### Y = A + Bx
# 
# Where Y is sales, x is the advertisement expenditure, A and B are constant.
# 
# After translating the hypothesis into the function, the basis for this is to find the relationship between the dependent and independent variables. The value of dependent variable is of most importance to researchers and depends on the value of other variables. Independent variable is used to explain the variation in the dependent variable. It can be classified into two types −
# 
# 1. Simple regression − One independent variable
# 
# 2. Multiple regression − Several independent variables
# 
# 1. Simple Regression
# 
# *Following are the steps to build up regression analysis −*
# 
# - Specify the regression model
# - Obtain data on variables
# - Estimate the quantitative relationships
# - Test the statistical significance of the results
# - Usage of results in decision-making
# - Formula for simple regression is −
# 
# #### Y = a + bX + u
# 
# - Y= dependent variable
# - X= independent variable
# - a= intercept
# - b= slope
# - u= random factor
# 
# Cross sectional data provides information on a group of entities at a given time, whereas time series data provides information on one entity over time. When we estimate regression equation it involves the process of finding out the best linear relationship between the dependent and the independent variables.
# 
# ### Method of Ordinary Least Squares (OLS)
# Ordinary least square method is designed to fit a line through a scatter of points is such a way that the sum of the squared deviations of the points from the line is minimized. It is a statistical method. Usually Software packages perform OLS estimation.
# 
# #### Y = a + bX
# 
# ### Co-efficient of Determination (R2)
# Co-efficient of determination is a measure which indicates the percentage of the variation in the dependent variable is due to the variations in the independent variables. R2 is a measure of the goodness of fit model. Following are the methods −
# 
# ### Total Sum of Squares (TSS)
# Sum of the squared deviations of the sample values of Y from the mean of Y.
# 
# #### TSS = SUM ( Yi − Y)2
# 
# - Yi = dependent variables
# - Y = mean of dependent variables
# - i = number of observations
# 
# ### Regression Sum of Squares (RSS)
# Sum of the squared deviations of the estimated values of Y from the mean of Y.
# 
# #### RSS = SUM ( Ỷi − uY)2
# 
# - Ỷi = estimated value of Y
# - Y = mean of dependent variables
# - i = number of variations
# 
# ### Error Sum of Squares (ESS)
# Sum of the squared deviations of the sample values of Y from the estimated values of Y.
# 
# #### ESS = SUM ( Yi − Ỷi)2
# 
# - Ỷi = estimated value of Y
# - Yi = dependent variables
# - i = number of observations
# 
# ![error_sum_of_squares.jpg](attachment:error_sum_of_squares.jpg)
# 
# ### What is R Squared?
# ​
# R2 measures the proportion of the total deviation of Y from its mean which is explained by the regression model. The closer the R2 is to unity, the greater the explanatory power of the regression equation. An R2 close to 0 indicates that the regression equation will have very little explanatory power.
# ​
# #### R2 = RSS/TSS= 1 - ESS/TSS
# 

# ![regression_equation.jpg](attachment:regression_equation.jpg)

# **Also notice in the code below how adjusted r squared penalises for irrelevant or useless independent variables**

# ###  Types of Regressions:
# * Linear Regression
# * Polynomial Regression
# * Logistic Regression
# * Quantile Regression
# * Ridge Regression
# * Lasso Regression
# * ElasticNet Regression
# * Principal Component Regression
# * Partial Least Square Regression
# * Support Vector Regression
# * Ordinal Regression
# * Poisson Regression
# * Negative Binomial Regression
# * Quasi-Poisson Regression
# * Cox Regression

# In[69]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model


# In[70]:


x_train , x_test , y_train , y_test = train_test_split(x,y , test_size= 0.3, random_state=4231)


# In[71]:


x_train.shape , x_test.shape ,y_train.shape, y_test.shape 


# In[72]:


compa={'train accuracy':[],'test accuracy':[],"Mean absolute error":[],'Mean absolute percentage error':[],
      'Mean squared error':[],'R Squared':[],'Adjusted R Squared':[]}


# ## Linear Regression

# In[73]:


reg = linear_model.LinearRegression()
reg.fit(x_train , y_train)
y_pred =reg.predict(x_test)

tra_acc=reg.score(x_train , y_train)*100
tes_acc=reg.score(x_test,y_test)*100
mae=mean_absolute_error(y_test,y_pred)
mape=mean_absolute_percentage_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)

n=x_test.shape[0]
p=x_test.shape[1] - 1
adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print("train accuracy: "+ str(tra_acc) + "%")
print("test accuracy: "+ str(tes_acc) + "%")
print("Mean absolute error: {}".format(mae))
print("Mean absolute percentage error: {}".format(mape))
print("Mean squared error: {}".format(mse))
print('R Squared: {}'.format(R2))
print('Adjusted R Squared: {}'.format(adj_rsquared))

compa['train accuracy'].append(tra_acc)
compa['test accuracy'].append(tes_acc)
compa["Mean absolute error"].append(mae)
compa['Mean absolute percentage error'].append(mape)
compa['Mean squared error'].append(mse)
compa['R Squared'].append(R2)
compa['Adjusted R Squared'].append(adj_rsquared)


# In[ ]:


df = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
df.sample(10)


# In[ ]:


pd.DataFrame(reg.coef_ , data_.columns[:-1] ,  columns=['Coeficient'])


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(df[50:150])
plt.legend(["Actual" , "Predicted"])
plt.show()


# ## Ridge and Lasso Regression

# **Ridge and Lasso regression are powerful techniques generally used for creating parsimonious models in presence of a ‘large’ number of features.**
# 
# #### Here ‘large’ can typically mean either of two things:
# 
# 1. Large enough to enhance the tendency of a model to overfit (as low as 10 variables might cause overfitting)
# 2. Large enough to cause computational challenges. With modern systems, this situation might arise in case of millions or billions of features
# Though Ridge and Lasso might appear to work towards a common goal, the inherent properties and practical use cases differ substantially. If you’ve heard of them before, you must know that they work by penalizing the magnitude of coefficients of features along with minimizing the error between predicted and actual observations. These are called ‘regularization’ techniques. The key difference is in how they assign penalty to the coefficients:
# 
# ### Ridge Regression:
# Performs L2 regularization, i.e. adds penalty equivalent to square of the magnitude of coefficients
# Minimization objective = LS Obj + α * (sum of square of coefficients)
# ### Lasso Regression:
# Performs L1 regularization, i.e. adds penalty equivalent to absolute value of the magnitude of coefficients
# Minimization objective = LS Obj + α * (sum of absolute value of coefficients)
# Note that here ‘LS Obj’ refers to ‘least squares objective’, i.e. the linear regression objective without regularization.
# 
# #### For Further Reference:[link here](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)

# ## Ridge regression

# In[ ]:


rid = linear_model.Ridge()
rid.fit(x_train , y_train)
y_pred =rid.predict(x_test)

tra_acc=rid.score(x_train , y_train)*100
tes_acc=rid.score(x_test,y_test)*100
mae=mean_absolute_error(y_test,y_pred)
mape=mean_absolute_percentage_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)

n=x_test.shape[0]
p=x_test.shape[1] - 1
adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print("train accuracy: "+ str(tra_acc) + "%")
print("test accuracy: "+ str(tes_acc) + "%")
print("Mean absolute error: {}".format(mae))
print("Mean absolute percentage error: {}".format(mape))
print("Mean squared error: {}".format(mse))
print('R Squared: {}'.format(R2))
print('Adjusted R Squared: {}'.format(adj_rsquared))

compa['train accuracy'].append(tra_acc)
compa['test accuracy'].append(tes_acc)
compa["Mean absolute error"].append(mae)
compa['Mean absolute percentage error'].append(mape)
compa['Mean squared error'].append(mse)
compa['R Squared'].append(R2)
compa['Adjusted R Squared'].append(adj_rsquared)


# In[ ]:


pd.DataFrame(rid.coef_ , data_.columns[:-1] ,  columns=['Coeficient'])


# In[ ]:


df = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
df.sample(10)


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(df[50:100])
plt.legend(["Actual" , "Predicted"])
plt.show()


# ## Lasso Regression

# In[ ]:


las = linear_model.Lasso()
las.fit(x_train , y_train)
y_pred =las.predict(x_test)

tra_acc=las.score(x_train , y_train)*100
tes_acc=las.score(x_test,y_test)*100
mae=mean_absolute_error(y_test,y_pred)
mape=mean_absolute_percentage_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)

n=x_test.shape[0]
p=x_test.shape[1] - 1
adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print("train accuracy: "+ str(tra_acc) + "%")
print("test accuracy: "+ str(tes_acc) + "%")
print("Mean absolute error: {}".format(mae))
print("Mean absolute percentage error: {}".format(mape))
print("Mean squared error: {}".format(mse))
print('R Squared: {}'.format(R2))
print('Adjusted R Squared: {}'.format(adj_rsquared))

compa['train accuracy'].append(tra_acc)
compa['test accuracy'].append(tes_acc)
compa["Mean absolute error"].append(mae)
compa['Mean absolute percentage error'].append(mape)
compa['Mean squared error'].append(mse)
compa['R Squared'].append(R2)
compa['Adjusted R Squared'].append(adj_rsquared)


# In[ ]:


pd.DataFrame(las.coef_ , data_.columns[:-1] ,  columns=['Coeficient'])


# In[ ]:


df = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
df.sample(10)


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(df[50:100])
plt.legend(["Actual" , "Predicted"])
plt.show()


# ## compersion between models

# In[ ]:


compa_df=pd.DataFrame(compa,index=["Linear Regression","Ridge regression","Lasso Regression"])
compa_df
