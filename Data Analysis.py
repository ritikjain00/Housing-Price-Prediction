#Importing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading Data
train_data = 'housing train data.csv'
test_data = 'housing test data.csv'
train = pd.read_csv(train_data)
test = pd.read_csv(test_data)

#Checking data
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))

#Checking NUll values
print(train.isnull().sum())
print(test.isnull().sum())

#Missing value of counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss>0]
miss.sort_values(inplace=True)
print(miss)

#Explaining missing values using- a bar plot
#Visualising missing values
miss = miss.to_frame()
miss.columns=['count']
miss.index.names = ['Name']
miss['Name']=miss.index

#plot the missing value count
sns.set(style='whitegrid', color_codes=True)
sns.barplot(x='Name',y='count',data=miss)
plt.xticks(rotation = 90)
plt.show()

#Checking distributin of target variable
sns.distplot(train['SalePrice'])

#Separate variable into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cate_data = train.select_dtypes(exclude=[np.number])
print('There are {} numeric and {} categorical columns'.format(numeric_data.shape[1],cate_data.shape[1]))

#Correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)

#top 15 values
print(corr['SalePrice'].sort_values(ascending=False)[:15],'\n')
#last 5 values
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#joinplot
sns.jointplot(x=train['GrLivArea'],y=train['SalePrice'])
sns.jointplot(x=train['OverallQual'],y=train['SalePrice'])

#In these we will remove inconsistency in GrLivArea
train.drop(train[train['GrLivArea'] > 4000].index,inplace=True)
train.reset_index(drop=True, inplace=True)

#now lets create heatmap for top 10 correlated features
cols =corr['SalePrice'].head(10).index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()






















