#Importing data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Reading Data
train_data = 'housing train data.csv'
test_data = 'housing test data.csv'
train = pd.read_csv(train_data)
test = pd.read_csv(test_data)

#Since we have already done the analysis in "data analysis.py"
#So here we will do feature engineering and model training,evaluation and testing

#Adding new features in train dataset
#Some feature are highly correlated with our target quantity i.e. SalePrice
train['GrLivArea_2']=train['GrLivArea']**2
train['GrLivArea_3']=train['GrLivArea']**3
train['GrLivArea_4']=train['GrLivArea']**4

train['TotalBsmtSF_2']=train['TotalBsmtSF']**2
train['TotalBsmtSF_3']=train['TotalBsmtSF']**3
train['TotalBsmtSF_4']=train['TotalBsmtSF']**4

train['GarageCars_2']=train['GarageCars']**2
train['GarageCars_3']=train['GarageCars']**3
train['GarageCars_4']=train['GarageCars']**4

train['1stFlrSF_2']=train['1stFlrSF']**2
train['1stFlrSF_3']=train['1stFlrSF']**3
train['1stFlrSF_4']=train['1stFlrSF']**4

train['GarageArea_2']=train['GarageArea']**2
train['GarageArea_3']=train['GarageArea']**3
train['GarageArea_4']=train['GarageArea']**4

train['BsmtFinSF1_2']=train['BsmtFinSF1']**2 
train['BsmtFinSF1_3']=train['BsmtFinSF1']**3
train['BsmtFinSF1_4']=train['BsmtFinSF1']**4 

train['BsmtUnfSF_2']=train['BsmtUnfSF']**2 
train['BsmtUnfSF_3']=train['BsmtUnfSF']**3
train['BsmtUnfSF_4']=train['BsmtUnfSF']**4 

train['TotRmsAbvGrd_2']=train['TotRmsAbvGrd']**2 
train['TotRmsAbvGrd_3']=train['TotRmsAbvGrd']**3
train['TotRmsAbvGrd_4']=train['TotRmsAbvGrd']**4 


#lets add 1stFlrSF and 2ndFlrSF and create new feature floorfeet
train['Floorfeet' ]= train['1stFlrSF'] + train['2ndFlrSF']
train = train.drop(['1stFlrSF','2ndFlrSF'],1)


#Adding new features in test dataset 
#Adding same features as that of train dataset so that no shape problem occurs
test['GrLivArea_2']=test['GrLivArea']**2
test['GrLivArea_3']=test['GrLivArea']**3
test['GrLivArea_4']=test['GrLivArea']**4

test['TotalBsmtSF_2']=test['TotalBsmtSF']**2
test['TotalBsmtSF_3']=test['TotalBsmtSF']**3
test['TotalBsmtSF_4']=test['TotalBsmtSF']**4

test['GarageCars_2']=test['GarageCars']**2
test['GarageCars_3']=test['GarageCars']**3
test['GarageCars_4']=test['GarageCars']**4

test['1stFlrSF_2']=test['1stFlrSF']**2 
test['1stFlrSF_3']=test['1stFlrSF']**3
test['1stFlrSF_4']=test['1stFlrSF']**4

test['GarageArea_2']=test['GarageArea']**2
test['GarageArea_3']=test['GarageArea']**3
test['GarageArea_4']=test['GarageArea']**4

test['BsmtFinSF1_2']=test['BsmtFinSF1']**2 
test['BsmtFinSF1_3']=test['BsmtFinSF1']**3
test['BsmtFinSF1_4']=test['BsmtFinSF1']**4 

test['BsmtUnfSF_2']=test['BsmtUnfSF']**2 
test['BsmtUnfSF_3']=test['BsmtUnfSF']**3
test['BsmtUnfSF_4']=test['BsmtUnfSF']**4 

test['TotRmsAbvGrd_2']=test['TotRmsAbvGrd']**2 
test['TotRmsAbvGrd_3']=test['TotRmsAbvGrd']**3
test['TotRmsAbvGrd_4']=test['TotRmsAbvGrd']**4 


#lets add 1stFlrSF and 2ndFlrSF and create new feature floorfeet
test['Floorfeet' ]= test['1stFlrSF'] + test['2ndFlrSF']
test = test.drop(['1stFlrSF','2ndFlrSF'],1)


#Copying the Id column from test dataset
test_id = test['Id'].copy()


# Remove rows with missing target, separate target from predictors
train.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train.SalePrice
train.drop(['SalePrice'], axis=1, inplace=True)


#Checking the Shape of dataset
print(train.shape)
print(test.shape)


#Separating data in training and validation
x_train, x_val, y_train, y_val = train_test_split(train, y,train_size=0.8, test_size=0.2,random_state=0)


#splitting data in numerical and categorical columns
#"Cardinality" means the number of unique values in a column
#Hence selecting categorical columns with relatively low cardinality 
cate_col = [cname for cname in x_train.columns if
                    x_train[cname].nunique() < 10 and 
                    x_train[cname].dtype == "object"]


# Select numerical columns
num_col = [cname for cname in x_train.columns if 
                   x_train[cname].dtype in ['int64', 'float64']]


#Keeping selected columns only
my_col = cate_col + num_col
x_train_full = x_train[my_col].copy()
x_validation = x_val[my_col].copy()
x_test = test[my_col].copy()


#Cleaning the numerical data
num_tf = SimpleImputer(strategy='median')
#Cleaning the categorical data
cate_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


#Handle unknown is set to ignore because sometimes in test set we have variables that were not present in Training set and hence were not encoded while training 
# but if we use these variables while testing we will get error hence to ignore these errors we use this argument

#Bundle preprocessing for numerical and categorical data
cleaning = ColumnTransformer(
    transformers=[
        ('num', num_tf, num_col),
        ('cat', cate_tf, cate_col)
    ])



from xgboost import XGBRegressor as xgbr 

model_1 = xgbr(random_state=42,n_estimators=2000,learning_rate=0.055) # Your code here

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('cleaning', cleaning),
                      ('model_1', model_1)
                     ])

# Preprocessing of training data, fit model 
clf.fit(x_train_full, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(x_validation)
print('MAE:', mean_absolute_error(y_val, preds))

#MAE score was found out to be 16839.557068707192 for XGB regressor
#Many other models were also checked but XGBRegressor performance was good
preds_test = clf.predict(x_test) 

# Save test predictions to file
output = pd.DataFrame({'Id': test_id,
                       'SalePrice': preds_test})
output.to_csv('submission_housing_xgb_4.csv', index=False)
