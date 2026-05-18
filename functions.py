
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
# Data Modeling Functions
# ----------------------------------------------------------------------------------------------------------------------------------

# Applies polynomial and ridge regressions to input data using time series cross-validation
# pol_degree = Degree of the regression (applies for polynomial and ridge). Random forest doesn't use it (use any value for that one)
# X: x-axis data (time)
# y: y-axis data (in this case, greenhouse gas component data)
# nsplits: number of splits on the dataset for the cross validation
# array with the regression types to be implemented (polynomial, ridge or random forest)
# Note: Currently, only applies for polynomial and ridge regressions, could be expanded for others.
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
    return [regression_label, np.mean(train_r2), np.mean(pred_r2)]

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
    return [regression_label, r2_score(y, model), np.sqrt(mean_squared_error(y, model))]

def apply_regressions(regressions, degrees, df, component):

    regression_metrics = []

    # Iterate the list of regressions to implement
    for regression in regressions:

        # Polynomial and Ridge regressions with degrees 1-3
        if(regression=="polynomial" or regression=="ridge"):
            for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                metrics = regression_model(degree, regression, df, component)
                regression_metrics.extend(np.round(metrics[1:3],4))

        # Random Forest regression
        elif(regression=="random_forest"):
            metrics = regression_model(1, regression, df, component)
            regression_metrics.extend(np.round(metrics[1:3],4))

    return regression_metrics

def apply_regressions_tscv(regressions, degrees, cv_slices, df, component):

    regression_metrics = []

    for regression in regressions:

        if(regression=="polynomial" or regression=="ridge"):
            for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                metrics = regression_predict_tscv(degree, cv_slices, regression, df, component)
                regression_metrics.extend(np.round(metrics[1:5],4))

        elif(regression=="random_forest"):
            metrics = regression_predict_tscv(degree, cv_slices, regression, df, component)
            regression_metrics.extend(np.round(metrics[1:5],4))

    return regression_metrics

# ----------------------------------------------------------------------------------------------------------------------------------
# Print functions
# ----------------------------------------------------------------------------------------------------------------------------------

def print_tscv_results(text, results):
    print("\n--------------------------------------------------")
    print(text + " Error Metrics:")
    print("--------------------------------------------------")
    print("Regression: " + results["regression"])
    print("R2 mean: " + str(results["r2_mean"].round(3)))
    print("R2 std.dev: " + str(results["r2_std"].round(3)))
    print("RMSE mean: " + str(results["rmse_mean"].round(3)))
    print("RMSE std.dev: " + str(results["rmse_std"].round(3)))
    print("MAE mean: " + str(results["mae_mean"].round(3)))
    print("MAE std.dev: " + str(results["mae_std"].round(3)))
