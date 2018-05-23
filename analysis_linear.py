import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns

import edatools

datafolder = 'data/working/'
outfolder = 'data/output/'
df_train = pd.read_csv(datafolder + 'train_normalized.csv')
df_test = pd.read_csv(datafolder + 'test_normalized.csv')

#categoricals = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour',
#                'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
#                'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual',
#                'BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual',
#                'GarageType', 'GarageFinish']
#linear_vars = ['GrLivArea', 'FullBath', 'GarageArea']
#poly_vars = ['OverallQual', 'TotalBsmtSF']

Xtrain = df_train.drop(['SalePrice','Id'], axis=1)
Xtest = df_test.drop(['Id'], axis=1)
y = np.log1p(df_train['SalePrice'])

model_linear = LinearRegression(normalize=True).fit(Xtrain, y)
model_lasso = Lasso(alpha=.0001, normalize=True, max_iter=1e5)
model_ridge = Ridge(alpha=.1, normalize=True, max_iter=1e5)
yfit = model_linear.predict(Xtest)

#sns.distplot(y, hist=False)
#sns.distplot(yfit, hist=False)
#plt.show()

randstate = 3
scores = cross_val_score(model_ridge,
                         Xtrain.sample(frac=1, random_state=randstate),
                         y.sample(frac=1, random_state=randstate),
                         cv=5,
                         scoring='neg_mean_squared_log_error')

#model_ridgecv = RidgeCV(alphas=[.0001, .001, .01, .1, 1.0, 10.0],
#				 		normalize=True,
#                 		cv=5,
#                 		scoring='neg_mean_squared_log_error')

#model_ridgecv.fit(Xtrain.sample(frac=1, random_state=randstate),
#                  y.sample(frac=1, random_state=randstate))

model_ridge.fit(Xtrain, y)
predictions = model_ridge.predict(Xtest)

submission = pd.DataFrame({'Id':df_test.Id, 'SalePrice':np.expm1(predictions)})
submission.to_csv(outfolder + 'submission3_ridge_normalized.csv', index=False)