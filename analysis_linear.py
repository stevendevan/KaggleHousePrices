import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

import edatools


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Remove 2 outliers as recommended by
# http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt
df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) &
                                  (df_train['SalePrice'] < 300000)].index)
ntrain = df_train.shape[0]

y = df_train['SalePrice']
df_train.drop(['SalePrice'], axis=1, inplace=True)
df_all = pd.concat([df_train, df_test], ignore_index=True)

df_all = edatools.condition_housing_data(df_all)

categoricals = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour',
                'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual',
                'BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual',
                'GarageType', 'GarageFinish']
linear_vars = ['GrLivArea', 'FullBath', 'GarageArea']
poly_vars = ['OverallQual', 'TotalBsmtSF']

df_categorical = pd.get_dummies(df_all[categoricals], drop_first=True)
df_poly = edatools.fill_poly_features(df_all[poly_vars], 2)
df_linear = df_all[linear_vars]

df_all_clean = pd.concat([df_categorical, df_poly, df_linear], axis=1)
Xtrain = df_all_clean[:ntrain]
Xtest = df_all_clean[ntrain:]


model_linear = LinearRegression(normalize=True).fit(Xtrain, y)
model_lasso = Lasso(alpha=.0001, normalize=True, max_iter=1e5)
model_ridge = Ridge(alpha=.001, normalize=True, max_iter=1e5)
yfit = model_linear.predict(Xtest)

sns.distplot(y, hist=False)
sns.distplot(yfit, hist=False)
plt.show()

randstate = 1
scores = cross_val_score(model_ridge,
                         Xtrain.sample(frac=1, random_state=randstate),
                         y.sample(frac=1, random_state=randstate),
                         cv=5,
                         scoring='neg_mean_squared_log_error')

model_ridge.fit(Xtrain, y)
predictions = model_ridge.predict(Xtest)

submission = pd.DataFrame({'Id':df_test.Id, 'SalePrice':predictions})
submission.to_csv('submission1_ridge.csv', index=False)