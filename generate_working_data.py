import pandas as pd
import pickle as pkl

import edatools

infolder = 'data/input/'
outfolder = 'data/working/'
df_train = pd.read_csv(infolder + 'train.csv')
df_test = pd.read_csv(infolder + 'test.csv')

# Remove 2 outliers as recommended by
# http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt

df_train.drop(df_train[(df_train['GrLivArea'] > 4000) &
                       (df_train['SalePrice'] < 300000)].index,
              inplace=True)
df_train.reset_index(inplace=True, drop=True)
ntrain = df_train.shape[0]

y = df_train['SalePrice']
df_train.drop(['SalePrice'], axis=1, inplace=True)
df_all = pd.concat([df_train, df_test], ignore_index=True)

df_all = edatools.condition_housing_data(df_all)

df_train = df_all[:ntrain]
df_test = df_all[ntrain:].reset_index(drop=True)

df_train = pd.concat([df_train, y], axis=1)

df_train.to_csv(outfolder + 'train_cleaned_79features.csv', index=False)
df_test.to_csv(outfolder + 'test_cleaned_79features.csv', index=False)