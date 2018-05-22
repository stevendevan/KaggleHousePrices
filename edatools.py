import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


def explore_categorical(df):

    plt.rc('figure', figsize=(10.0, 5.0))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # discard non-object data
    df_ob = df.loc[:, df.dtypes == 'object']
    df_ob.fillna('none', inplace=True)

    for column in df_ob.columns:

        values = df_ob[column].groupby(
            df_ob[column]).count().sort_values(ascending=False)

        fig1, (ax11, ax12) = plt.subplots(1, 2)
        ax11.bar(values.index, values.values, color=colors)
        plt.title(column)

        for label in values.index:
            data = df.loc[df_ob[column] == label, 'SalePrice'].values
            if len(data) > 1:
                sns.distplot(data, hist=False, ax=ax12)

        plt.show()


def explore_numerical(df):

    df_num = df.loc[:, df.dtypes != 'object']
    df_num.fillna(0, inplace=True)
    df_num.drop(labels='Id', axis=1, inplace=True)

    for column in df_num.columns:
        plt.scatter(df_num[column].values, df_num['SalePrice'].values,
                    alpha=0.4, edgecolors='none')
        plt.title(column)
        plt.show()


def fill_poly_features(df, power):

    poly_model = sklearn.preprocessing.PolynomialFeatures(
        power, include_bias=False)

    for colname in df.columns:
        new_features = poly_model.fit_transform(df[colname].values[:, None])
        df_new = pd.DataFrame(new_features[:, -1], columns=[colname + '^2'])
        df = pd.concat([df, df_new], axis=1)

    return df


def condition_housing_data(df):
    """General data-conditioning function to prepare the housing DataFrame for
    analysis. Mostly NaN filling
    """

    fillnone = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
                'MasVnrType']

    fillzero = ['GarageArea', 'TotalBsmtSF', 'LotFrontage', 'MasVnrArea',
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

    fillmode = ['Electrical', 'KitchenQual', 'SaleType', 'Exterior1st',
                'Exterior2nd', 'Functional', 'MasVnrType', 'MSZoning']

    # has some NaNs. Value is highly correlated with YearBuilt
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    # There seems to be an erroneous value for GarageYrBlt of 2207
    # Based on the YearBuilt being 2006, I assume it should be 2007
    df.loc[df.GarageYrBlt == 2207.0, 'GarageYrBlt'] = 2007.0

    # Convert column to strings. It's categorical data stored as int64
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Really only one value present
    df.drop(['Utilities'], axis=1, inplace=True)

    # Apparently this can't be done without looping.
    for colname in fillnone:
        df[colname].fillna('none', inplace=True)

    for colname in fillzero:
        df[colname].fillna(0, inplace=True)

    for colname in fillmode:
        df[colname].fillna(df[colname].mode()[0], inplace=True)

    return df
