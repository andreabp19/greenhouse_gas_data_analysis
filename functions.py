
# Author: Andrea Pineda
# Date: 11 May. 2026
# Summary: Custom functions used in main.py
# Last modified: 11 May. 2026

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    r2_results = []
    rmse_results = []
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
            train = LinearRegression().fit(X_train, y_train) # Train model
            prediction = train.predict(X_test) # Predict data
            regression_label = regression_type+str(pol_degree)

        # 2. Ridge regression
        elif (regression_type=="ridge"):
            train = Ridge(alpha=0.0001).fit(X_train, y_train) # Train model
            prediction = train.predict(X_test) # Predict data
            regression_label = regression_type+str(pol_degree)

        # 3. Random Forest regressor
        elif (regression_type=="random_forest"):
            train = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train, y_train) # Train model
            prediction = train.predict(X_test) # Predict data
            regression_label = regression_type

        # Compute error metrics: R2, RMSE, MAE
        r2_results.append(r2_score(y_test, prediction)) # Compute and store R2
        rmse_results.append(np.sqrt(mean_squared_error(y_test, prediction))) # Compute and store RMSE
        mae_results.append(mean_absolute_error(y_test, prediction)) # Compute and store MAE

    # Return a dictionary with the error metrics for the applied regression
    return {
    "regression": regression_label,
    "r2_mean": np.mean(r2_results),
    "r2_std": np.std(r2_results),
    "rmse_mean": np.mean(rmse_results),
    "rmse_std": np.std(rmse_results),
    "mae_mean": np.mean(mae_results),
    "mae_std": np.std(mae_results)
    }

def select_best_fit(regression_metrics):

    r2_means = {}
    rmse_means = {}
    
    for regression in regression_metrics:
        r2_means[regression["regression"]] = regression["r2_mean"]
        rmse_means[regression["regression"]] = regression["rmse_mean"]

    min_r2_means = max(r2_means.values())
    min_rmse_means = min(rmse_means.values())

    for key in r2_means.keys():
        if min_r2_means == r2_means[key]:
            best_r2 = ["R2", key, min_r2_means];

        if min_rmse_means == rmse_means[key]:
            best_rmse = ["RMSE", key, min_rmse_means]

    return best_r2, best_rmse



def regression_model(components_to_predict, countries_in_dataset, df):

    results_table = ["Country", "Linear R2", "Linear RMSE", "Quadratic R2", "Quadratic RMSE"]

    historical_modeling_table = []

    # Select a component from the list of greenhouse gas components to model
    for component in components_to_predict:

        # Add a row with info of the current component
        historical_modeling_table.append([])
        historical_modeling_table.append([component])
        historical_modeling_table.append(results_table)

        # Select a country from which to model the data
        for country in countries_in_dataset:

            # Empty array to save the results for the current country
            country_regression_metrics = [] 
            country_regression_metrics.append(country)

            # Retrieve data for the current country
            df_country = df.loc[country]

            # y = CO2 values, X = time (years in the dataset)
            y = df_country[component].to_numpy().reshape(-1, 1) # Reshape is necessary because it's 1D and the necessary functions expect a 2D array
            X = df_country.index.to_numpy().reshape(-1, 1) # Same reshape to 2D

            # Linear Regression Model
            lin_model = LinearRegression().fit(X, y) # Generate linear regression model
            y_lin = lin_model.predict(X) # Predict CO2 values based on the model

            # Quadratic Regression Model
            poly2 = PolynomialFeatures(degree=2) # Generate quadratic polynomial features
            X_quad = poly2.fit_transform(X) # Fit quadratic features to the time data for use in the model

            quad_model = LinearRegression().fit(X_quad, y) # Generate quadratic model based on Linear Regression
            y_quad = quad_model.predict(X_quad) # Predict CO2 values based on the quadratic model

            # Linear model error metrics: R2 and RMSE - Generate and save into the list for the current country
            country_regression_metrics.append(r2_score(y, y_lin))
            country_regression_metrics.append(np.sqrt(mean_squared_error(y, y_lin)))

            # Quadratic model error metrics: R2 and RMSE - Generate and save into the list for the current country
            country_regression_metrics.append(r2_score(y, y_quad))
            country_regression_metrics.append(np.sqrt(mean_squared_error(y, y_quad)))

            # Append data for the current country into the component table
            historical_modeling_table.append(country_regression_metrics)

    return historical_modeling_table

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
