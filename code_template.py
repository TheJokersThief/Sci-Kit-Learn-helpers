# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Error Estimation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

# Feature Selection
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from math import floor, ceil

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data exploration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def describe_data(data):
    print("SHAPE: ")
    print(data.shape)
    print("\n\n")

    print("COLUMNS: ")
    print(data.columns)
    print("\n\n")

    print("DATATYPES: ")
    print(data.dtypes)
    print("\n\n")


def print_nan_examples(data):
    nanCols = data.isnull().any()
    columns = data.columns[nanCols]
    [ print( "COLUMN " + column + ": ", data[column].isnull().sum() ) for column in columns ]


def scatter_plot(data, x_col, y_col):
    x = data[ [x_col] ].values
    y = data[ [y_col] ].values
    fig = plt.figure()
    plt.title( x_col + " against " + y_col )
    plt.xlabel( x_col )
    plt.ylabel( y_col )
    plt.scatter(x, y, color = 'green')
    plt.show()

def histogram(data, class_col, plot_col, height = 20, step = 10):
    x = data[ [plot_col] ].values
    y = data[ [class_col] ].values
    fig = plt.figure(figsize=(10,4))
    classes = data[class_col].unique()
    minimum = floor(data[plot_col].min())
    maximum = ceil(data[plot_col].max())

    colours = matplotlib_colours()
    for counter in range(1,len(classes) + 1):
        plt.subplot(1, len(classes),counter)
        plt.title(classes[counter-1])
        plt.hist(x[y == classes[counter-1]], bins = range(minimum, maximum + step, step), color = colours[counter-1])
        plt.axis([minimum, maximum + step, 0, height])
    plt.show()

def boxplot(data, class_col, plot_col):
    x = data[ [plot_col] ].values
    y = data[ [class_col] ].values
    fig = plt.figure()
    ax = plt.axes()
    classes = data[class_col].unique()

    toplot = []
    labels = []
    for aclass in classes:
        toplot.append(x[y == aclass])
        labels.append(aclass)

    ax.set_xticklabels(labels)
    plt.boxplot(toplot)
    plt.show()
    return True

def frequencies(data, class_col, plot_col):
    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    sns.factorplot(plot_col, class_col, data = data,size=4,aspect=3)
    sns.countplot(x = plot_col, data = data, ax=axis1)
    sns.countplot(x = class_col, hue=plot_col, data = data, ax=axis2)
    embark_perc = data[[plot_col,  class_col]].groupby(df[plot_col],as_index=False).mean()
    sns.barplot(x = plot_col, y = class_col, data = embark_perc,ax=axis3)
    return sns.plt.show()

def distribution(data, class_col, plot_col, variance = 25):
    facet = sns.FacetGrid(data, hue = class_col,aspect=4)
    facet.map(sns.kdeplot, plot_col,shade= True)
    facet.set(xlim=(data[plot_col].min() - variance, data[plot_col].max() + variance))
    facet.add_legend()

    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    average = data[[ plot_col,  class_col]].groupby([ plot_col],as_index=False).mean()
    sns.barplot(x = plot_col, y = class_col, data = average)
    return sns.plt.show()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    return sns.plt.show()

def matplotlib_colours():
    return [
        'blue',
        'green',
        'red',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'white'
    ]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def replace_unknown_values(data, columns):
    return [ 
                data[column].replace({'?': np.nan, 'N/A': np.nan, 'UNK': np.nan}, inplace=True) 
                for column 
                    in columns
            ]

def cast_columns_float(data, columns):
    for column in columns:
        data[column] = data[column].astype(float)
    return True

def cast_columns_string(data, columns):
    for column in columns:
        data[column] = data[column].apply(str)
    return True

def dropna(data):
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nominal Data Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def one_hot_encode(data, column):
    dummies = pd.get_dummies(data[column], prefix=column)
    data = pd.concat([data, dummies], axis=1)
    data.drop(column, axis=1, inplace=True)
    return data


def binary_encode(data, column, zero, one):
    data.replace({ column : { zero : 0, one: 1 } }, inplace=True)
    return data

def print_full(content):
    pd.set_option('display.max_colwidth', -1)
    print(content)
    pd.reset_option('display.max_colwidth')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Error Estimation (Regression)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cross_val_method_LeaveOneOut():
    return LeaveOneOut()

def cross_val_method_KFold():
    return KFold( n_splits =10, shuffle = True )

def cross_val_method_ShuffleSplit():
    return ShuffleSplit(n_splits = 10, test_size = 0.3)

def regression_estimator_ols(stages = []):
    stages.extend([
                regression_standardisation(),               
                ('estimator', LinearRegression())
            ])

    return Pipeline(stages)

def regression_estimator_Lasso(stages = [], alpha = 0.3):
    stages.extend([
                regression_standardisation(),               
                ('estimator', Lasso( alpha = alpha ))
            ])

    return Pipeline(stages)

def regression_estimator_LassoCV(stages = []):
    stages.extend([
                regression_standardisation(),               
                ('estimator', LassoCV( cv = 10 ))
            ])

    return Pipeline(stages)

def regression_estimator_Ridge(stages = [], alpha = 0.3):
    stages.extend([
                regression_standardisation(),               
                ('estimator', Ridge( alpha = alpha ))
            ])

    return Pipeline(stages)

def regression_estimator_RidgeCV(stages = []):
    stages.extend([
                regression_standardisation(),               
                ('estimator', RidgeCV( cv = 10 ))
            ])

    return Pipeline(stages)

def regression_estimator_knn(stages = [], cross_val_method = 10, scoring = 'neg_mean_squared_error'):
    stages.extend([
                regression_standardisation(),               
                ('estimator', KNeighborsRegressor(weights = inv_distances))
            ])

    knn = Pipeline(stages)
    return GridSearchCV(knn, knn_parameters(), scoring = scoring, cv = cross_val_method)

def regression_estimators(stages = [], cross_val_method = 10, scoring = 'neg_mean_squared_error'):
    return {
        'OLS \t\t' : regression_estimator_ols(stages),
        'Lasso \t\t': regression_estimator_LassoCV(stages),
        'Ridge \t\t' : regression_estimator_RidgeCV(stages),
        'kNN (weighted)\t' : regression_estimator_knn(stages, cross_val_method, scoring),
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Error Estimation (Classification)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Error Estimation (Classification)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cross_val_StratifiedShuffleSplit():
    return StratifiedShuffleSplit(n_splits = 2, test_size = 0.3)

def cross_val_StratifiedKFold():
    return StratifiedKFold(n_splits = 0, shuffle = True)

def classification_estimator_LogisticRegression(stages1, cv):
    stages1.extend([
                get_standardisation(),               
                ('estimator', LogisticRegressionCV( Cs = log_reg_parameters(), penalty="l2", cv = cross_val_method))
            ])

    return Pipeline(stages1)

def classification_estimator_LogisticRegression_l1(stages2, cv):
    stages2.extend([
                get_standardisation(),               
                ('estimator', LogisticRegressionCV( Cs = log_reg_parameters(), penalty = 'l1', solver='liblinear', cv = cross_val_method ))
            ])

    return Pipeline(stages2)

def classification_estimator_KNeighborsClassifier(stages3 = [], cross_val_method = 10, scoring = 'accuracy'):
    stages3.extend([
                get_standardisation(),               
                ('estimator', KNeighborsClassifier(weights = inv_distances))
            ])

    knn = Pipeline(stages3)
    return GridSearchCV(knn, knn_parameters(), scoring = scoring, cv = cross_val_method)

def classification_estimators(stages = [], cross_val_method = 10, scoring = 'accuracy'):
    stages1 = stages.copy()
    stages2 = stages.copy()
    stages3 = stages.copy()
    
    return {
        'l1' : classification_estimator_LogisticRegression_l1(stages1, cross_val_method),
        'l2' : classification_estimator_LogisticRegression(stages2, cross_val_method),
#         'kNN' : classification_estimator_KNeighborsClassifier(stages3, cross_val_method),
    }

def plot_learning_curve(estimator, X, y, scoring, cv):
    train_set_sizes = np.linspace(.1, 1.0, 10)
    train_sizes, mses_train, mses_test = learning_curve(estimator, X, y, train_sizes=train_set_sizes, cv=cv, scoring=scoring)
    mean_mses_train = np.mean(np.abs(mses_train), axis=1)
    mean_mses_test = np.mean(np.abs(mses_test), axis=1)

    fig = plt.figure()
    plt.xlabel("num. training examples")
    plt.ylabel(scoring)
    plt.ylim(0, 10)
    plt.plot(train_sizes, mean_mses_train, label = 'training ' + scoring, color = 'purple')
    plt.plot(train_sizes, mean_mses_test, label='test ' + scoring, color = 'orange')
    plt.legend()
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Feature Engineering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def higher_order(data, column, max_order):
    values = data[column].values
    powered_values = values
    for x in xrange(1, max_order + 1):
        powered_values *= values
        data[column + str(x)] = powered_values
    return data

def find_one(substrs, superstr):
    for substr in substrs:
        if superstr.find(substr) != -1:
            return substr
    return ''

def extract_strings_one_hot(data, column, new_col_title, search_values = []):
    name_values = data[column].values
    found_values = []
    for haystack in haystacks:
        found_values.append(find_one( search_values, haystack ) )
    
    one_hot = pd.get_dummies(found_values, new_col_title, '_')
    data.drop(column, axis=1, inplace=True)
    data = pd.concat([data, one_hot], axis=1)
    return data

def add_columns(data, col1, col2, new_col_title):
    col1_values = data[col1].values
    col2_values = data[col2].values
    result = col1_values + col2_values
    data[new_col_title] = result
    return data

def subtract_columns(data, col1, col2, new_col_title):
    col1_values = data[col1].values
    col2_values = data[col2].values
    result = col1_values - col2_values
    data[new_col_title] = result
    return data

def multiply_columns(data, col1, col2, new_col_title):
    col1_values = data[col1].values
    col2_values = data[col2].values
    result = col1_values * col2_values
    data[new_col_title] = result
    return data

def divide_columns(data, col1, col2, new_col_title):
    col1_values = data[col1].values
    col2_values = data[col2].values
    result = col1_values / col2_values
    data[new_col_title] = result
    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Feature Selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_support_features(X, y, estimator, cv, scoring):
    selector = RFECV(estimator, cv = cv, scoring = scoring)
    selector = selector.fit(X, y)
    return selector.get_support()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def regression_get_alpha(pipeline):
    if( isinstance(pipeline.named_steps['estimator'], Lasso) 
            or isinstance(pipeline.named_steps['estimator'], Ridge) ):
        return pipeline.named_steps['estimator'].alpha_
    
    return False

def get_GridCV_best_params(pipeline):
    if( isinstance(pipeline.named_steps['estimator'], KNeighborsClassifier) ):
        pipeline.named_steps['estimator'].best_params_['estimator__n_neighbors']
    elif( isinstance(pipeline.named_steps['estimator'], LogisticRegression) ):
        pipeline.named_steps['estimator'].best_params_['estimator__Cs']

    return False

def get_standardisation():
    return ('standardize', StandardScaler())

def get_imputer(name, missing_vals, strat):
    return ('impute_' + name, Imputer(missing_values=missing_vals, strategy=strat))

def knn_parameters():
    return { 'estimator__n_neighbors' : list(range(1,16)) }

def log_reg_parameters():
    values = list(range(1,16))
    values.append(sys.maxsize)
    return values

def cross_evaluate(estimators, X, y, cross_val_method, scoring_method):
    for name, estimator in estimators.items():
        mses_test = cross_val_score(estimator, X, y, scoring = scoring_method, cv = cross_val_method, n_jobs=-1)        
        mean_mse_test = np.mean(mses_test)
        print( name + " : " + str(mean_mse_test))
    return estimators


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def inv_distances(dists):
    return 1 / (0.0001 + dists)
