import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler


def explore_categorical(df):

    plt.rc('figure', figsize=(10.0, 5.0))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # discard non-object data
    df_ob = df.loc[:, df.dtypes == 'object'].fillna('none')

    for column in df_ob.columns:

        values = df_ob[column].groupby(
            df_ob[column]).count().sort_values(ascending=False)

        fig1, (ax11, ax12) = plt.subplots(1, 2)
        plt.suptitle(column)
        ax11.bar(values.index, values.values, color=colors)
        plt.title('Feature value counts')
        ax11.set_xlabel('Feature value')
        ax11.set_ylabel('Count')

        for label in values.index:

            data = df.loc[df_ob[column] == label, 'SalePrice'].values
            if len(data) > 1:
                sns.distplot(data, hist=False, ax=ax12)
                plt.title('PDF per feature-value')
                ax12.set_xlabel('SalePrice ($)')
                ax12.set_ylabel('Relative requency of occurance\n'
                                'Units are not that useful')
                ax12.set_yticks([])
                # Maybe consider CDF as an alternative
                # data.hist(bins=len(data), cumulative=True,
                #          density=True, histtype='step')

        plt.show()


def explore_numerical(df):

    df_num = df.loc[:, df.dtypes != 'object']
    df_num.fillna(0, inplace=True)
    df_num.drop(labels='Id', axis=1, inplace=True)

    for column in df_num.columns:
        plt.scatter(df_num[column].values, df_num['SalePrice'].values,
                    alpha=0.4, edgecolors='none')
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('SalePrice ($)')
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


def dummify_data(df):

    categoricals = df.columns[df.dtypes == 'object']
    df_new = df.drop(categoricals, axis=1)
    df_new = pd.concat([df_new, pd.get_dummies(df[categoricals])],
                       axis=1)

    return df_new


def normalize_data(df):

    numericals = df.columns[df.dtypes != 'object']
    skewness = df[numericals].apply(stats.skew)
    numericals_skewed = numericals[np.abs(skewness) > 2.0]
    df[numericals_skewed] = df[numericals_skewed].apply(np.log1p)

    return df
