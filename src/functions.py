
# Author: Andrea Pineda
# Date: 11 May. 2026
# Summary: Custom functions used in this project
# Last modified: 18 May. 2026

# ----------------------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------------------

RIDGE_ALPHA = 0.1

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Modeling and Prediction Functions
# ----------------------------------------------------------------------------------------------------------------------------------

# Brief: Applies a selected regression to data using time series cross-validation for training and prediction
# Arguments:
#   pol_degree:         Degree of the regression (applies for polynomial and ridge). Random Forest doesn't use it (choose any value).
#   X:                  x-axis data (time)
#   y:                  y-axis data (in this case, greenhouse gas component data)
#   nsplits:            Number of splits on the dataset for the cross validation
#   regression_type:    Array with the regression types to be implemented (currently supports: polynomial, ridge or random forest)
#   component:          Variable to be modeled
# Returns: R2 score of the regression
def regression_predict_tscv(pol_degree, nsplits, regression_type, df, component):

    # Arrays for saving the error metric results
    train_r2 = []
    pred_r2 = []
    mae_results = []

    X = df.index.to_numpy().reshape(-1,1) # Time (years)
    X_centered = X - X.min() # Center years around 0 instead of 1000+ dates
    
    tscv = TimeSeriesSplit(n_splits=nsplits) # Generate time series split for cross validation

    # 1. Generate cross validation subsets on X according to the type of regression
    if (regression_type=="polynomial" or regression_type=="ridge"):
        y = df[component].to_numpy().reshape(-1,1) # Greenhouse gas component percentage
        X_set = PolynomialFeatures(pol_degree).fit_transform(X_centered) # Generate polynomial features
        splits = list(tscv.split(X_set,y)) # Generate splits with the polynomial data
    
    elif (regression_type=="random_forest"):
        y = df[component].to_numpy() # Greenhouse gas component percentage
        splits = list(tscv.split(X_centered,y)) # Generate splits with non polynomial features
        X_set = X_centered # Save in X_set for easier code in the next part
    
    # 2. Apply the chosen regression with cross validation
    for train_idx, test_idx in splits:
        
        # Slice X and y data for training and test subsets
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X_set[train_idx,:], X_set[test_idx,:]

        # Apply model training and prediction:

        # 1. Polynomial regression
        if (regression_type=="polynomial"):
            fit = LinearRegression().fit(X_train, y_train) # Train model
            train_pred = fit.predict(X_train) # Prediction with training data
            test_pred = fit.predict(X_test) # Prediction with test data
            regression_label = regression_type+str(pol_degree)

        # 2. Ridge regression
        elif (regression_type=="ridge"):
            fit = Ridge(alpha=RIDGE_ALPHA).fit(X_train, y_train) # Train model
            train_pred = fit.predict(X_train) # Prediction with training data
            test_pred = fit.predict(X_test) # Prediction with test data
            regression_label = regression_type+str(pol_degree)

        # 3. Random Forest regressor
        elif (regression_type=="random_forest"):
            fit = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train, y_train) # Train model
            train_pred = fit.predict(X_train) # Prediction with training data
            test_pred = fit.predict(X_test) # Prediction with test data
            regression_label = regression_type

        # Compute R2 for training and prediction
        train_r2.append(r2_score(y_train, train_pred)) # Training prediction R2 score
        pred_r2.append(r2_score(y_test, test_pred)) # Test prediction R2 score

    # Return a list with the regression name and the R2 scores for training and prediction
    return [np.mean(train_r2), np.mean(pred_r2)]


# Brief: Applies a selected regression to data to fit historical data and returns the R2 score.
# Arguments:
#   pol_degree:         Degree of the regression (applies for polynomial and ridge). Random Forest doesn't use it (choose any value).
#   regression_type:    Array with the regression types to be implemented (currently supports: polynomial, ridge or random forest)
#   df:                 Dataframe from which to extract and predict data.
#   component:          Variable to be predicted
# Returns: R2 score of the regression
def regression_model(pol_degree, regression_type, df, component):

    # Retrieve X
    X = df.index.to_numpy().reshape(-1,1) # Time (years)
    X_centered = X - X.min() # Center years around 0 instead of 1000+ dates

    # 1. Retrieve X and y from the df
    if (regression_type=="polynomial" or regression_type=="ridge"):
        y = df[component].to_numpy().reshape(-1,1) # Greenhouse gas component percentage
        X_set = PolynomialFeatures(pol_degree).fit_transform(X_centered) # Generate polynomial features

    elif (regression_type=="random_forest"):
        y = df[component].to_numpy() # Greenhouse gas component percentage
        X_set = X_centered # Save in X_set for easier code in the next part

    # 2. Model historical data of the component
    if(regression_type=="polynomial"):
        fit = LinearRegression().fit(X_set, y) # Generate linear regression model
        model = fit.predict(X_set) # Model compoment data
        regression_label = regression_type+str(pol_degree)

    elif(regression_type=="ridge"):
        fit = Ridge(alpha=RIDGE_ALPHA).fit(X_set, y) # Generate ridge regresion
        model = fit.predict(X_set) # Model compoment data
        regression_label = regression_type+str(pol_degree)

    elif(regression_type=="random_forest"):
        fit = RandomForestRegressor(max_depth=2, random_state=0).fit(X_set, y) # Train model
        model = fit.predict(X_set) # Model compoment data
        regression_label = regression_type

    # Return a list with the error metrics for the regression
    return r2_score(y, model)

# Brief: Applies a list of regressions to fit historical data and returns the R2 score of each by using regression_model().
# Arguments:
#   regressions:    List of regressions to apply (currently supports polynomial, ridge and random forest)
#   degrees:        Degree of the regressions (applies for polynomial and ridge). Random Forest doesn't use it (choose any value).
#   df:             Dataframe from which to extract and model data
#   component:      Variable to be modeled
# Returns: A list with the R2 scores of the regressions
def apply_regressions(regressions, degrees, df, component):

    regression_metrics = []

    # Iterate the list of regressions to implement
    for regression in regressions:

        # Polynomial and Ridge regressions with degrees 1-3
        if(regression=="polynomial" or regression=="ridge"):
            for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                metrics = regression_model(degree, regression, df, component)
                regression_metrics.append(np.round(metrics,4))

        # Random Forest regression
        elif(regression=="random_forest"):
            metrics = regression_model(1, regression, df, component)
            regression_metrics.append(np.round(metrics,4))

    return regression_metrics

# Brief: Applies a list of regressions to predict data and returns the R2 score of each by using regression_predict_tscv().
# Arguments:
#   regressions:    List of regressions to apply (currently supports polynomial, ridge and random forest)
#   degrees:        Degree of the regressions (applies for polynomial and ridge). Random Forest doesn't use it (choose any value).
#   cs_slices:      Number of slices or divisions to implement on the input dataframe for cross-validation
#   df:             Dataframe from which to extract and model data
#   component:      Variable to be modeled
# Returns: A list with the R2 scores of the regressions
def apply_regressions_tscv(regressions, degrees, cv_slices, df, component):

    regression_metrics = []

    for regression in regressions:

        if(regression=="polynomial" or regression=="ridge"):
            for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                metrics = regression_predict_tscv(degree, cv_slices, regression, df, component)
                regression_metrics.extend(np.round(metrics,4))

        elif(regression=="random_forest"):
            metrics = regression_predict_tscv(degree, cv_slices, regression, df, component)
            regression_metrics.extend(np.round(metrics,4))

    return regression_metrics

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Postprocessing Functions
# ----------------------------------------------------------------------------------------------------------------------------------

# Brief: Takes an input table formatted as a list of lists [[...], [...], ...] and generates a draframe.
# Arguments:
#   raw_table:  List of lists containing the table to be converted to a dataframe
# Returns: The resulting dataframe after converting the table    
def df_from_raw(raw_table):
    new_df = pd.DataFrame(raw_table) # Generate DataFrame from the table [[...], [...], ...]
    new_df.columns = new_df.iloc[0] # Set the row with labels as column names
    new_df = new_df[1:].reset_index(drop=True) # Remove the row that has the number-filled column labels
    
    return new_df

# Brief: Takes an input dataframe and determines the best regression to fit each row of data
# Arguments:
#   df:     Input dataframe
# Returns: The input dataframe with additional columns for the name and R2 score of the best regression fit     
def get_best_regression_fit(df):
    df["Best Fit"] = df.iloc[:, 2:].idxmax(axis=1) # Label of the best regression fit
    df["Best R2"] = df.iloc[:, 2:-1].max(axis=1) # R2 score of the best regression (returns a Series)
    df = pd.DataFrame(df).reset_index(drop=True) # Convert from Series to Dataframe
    
    return df

# ----------------------------------------------------------------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------------------------------------------------------------

# Brief: Plots a stacked bar chart based on a given dataframe
# Arguments:
#   colors: List of colors to be used (in hexadecimal)
#   df:     Input dataframe to plot
#   title:  String with the desired title for the plot
# Returns: None.
def plot_regression_count(colors, df, title):
    
    ax = df.plot(kind='bar', stacked=True, figsize=(12,8), color=colors)

    # Add values into each nonzero stacked bar
    for container in ax.containers:

        labels = []

        for bar in container:

            height = bar.get_height()
            if (height != 0):
                labels.append(f"{int(height)}") # Save only the labels that aren't 0
                
            else:
                labels.append("")    

        ax.bar_label(container, labels=labels, label_type="center")
            
    # Plot config
    plt.title(title)
    plt.xlabel("Greenhouse Gas Component")
    plt.ylabel("Number of best fit cases with each regression type")
    plt.show()