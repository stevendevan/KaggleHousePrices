import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score

datafolder = 'data/working/'
outfolder = 'data/output/'
df_test = pd.read_csv(datafolder + 'test_normalized.csv')
df_train = pd.read_csv(datafolder + 'train_normalized.csv')

Xtrain = df_train.drop(['SalePrice','Id'], axis=1)
Xtest = df_test.drop(['Id'], axis=1)
y = np.log1p(df_train['SalePrice'])

model = DecisionTreeRegressor(min_samples_leaf=10,
							  max_depth=None)
adaboost = AdaBoostRegressor(model, n_estimators=500)

randstate = 2
#scores = cross_val_score(adaboost,
#                         Xtrain.sample(frac=1, random_state=randstate),
#                         y.sample(frac=1, random_state=randstate),
#                         cv=5,
#                         scoring='neg_mean_squared_log_error')

adaboost.fit(Xtrain, y)
predicted = adaboost.predict(Xtest)
submission = pd.DataFrame({'Id':df_test.Id, 'SalePrice':np.expm1(predicted)})
submission.to_csv(outfolder + 'submission4_adaboost_normalized.csv', index=False)