
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: Coursera data science project
# Subject: greenhouse gas dataset from the UN (obtained from Kaggle)
# Last modified: 9 May. 2026

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

from config import DATASET_PATH # From local file with the path to the dataset
from config import PROJECT_PATH # From local file with the path to the project's folder

# ----------------------------------------------------------------------------------------------------------------------------------
# Import dataset
# ----------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Configs for printing the data in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ----------------------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------------------

def polynomial_tscv(pol_degree, X, y, nsplits, regression_type):

    r2_results = []
    rmse_results = []
    mae_results = []

    X_centered = X - X.min() # Center years around 0 instead of 1000+ dates
    X_set = PolynomialFeatures(pol_degree).fit_transform(X_centered) # Polynomial features

    # Generate cross validation splits
    tscv = TimeSeriesSplit(n_splits=nsplits)
    splits = list(tscv.split(X_set,y))
    
    for train_idx, test_idx in splits:
        
        # Slice training and test subsets
        X_train, X_test = X_set[train_idx,:], X_set[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        # Model train and predict

        if (regression_type=="polynomial"):
            train = LinearRegression().fit(X_train, y_train)
            prediction = train.predict(X_test)

        elif (regression_type=="ridge"):
            train = Ridge(alpha=0.0001).fit(X_train, y_train)
            prediction = train.predict(X_test)

        # Compute error metrics
        r2_results.append(r2_score(y_test, prediction)) # Compute and store R2
        rmse_results.append(np.sqrt(mean_squared_error(y_test, prediction))) # Compute and store RMSE
        mae_results.append(mean_absolute_error(y_test, prediction)) # Compute and store MAE

    return {
    "regression": regression_type+str(pol_degree),
    "r2_mean": np.mean(r2_results),
    "r2_std": np.std(r2_results),
    "rmse_mean": np.mean(rmse_results),
    "rmse_std": np.std(rmse_results),
    "mae_mean": np.mean(mae_results),
    "mae_std": np.std(mae_results)
    }

def random_forest_regressor_tscv(X, y, nsplits):

    r2_results = []
    rmse_results = []
    mae_results = []

    X_centered = X - X.min() # Center years around 0 instead of 1000+ dates

    # Generate cross validation splits
    tscv = TimeSeriesSplit(n_splits=nsplits)
    splits = list(tscv.split(X_centered,y))
    
    for train_idx, test_idx in splits:
        
        # Slice training and test subsets
        X_train, X_test = X_centered[train_idx,:], X_centered[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        # Model train and predict
        train = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train, y_train)
        prediction = train.predict(X_test)

        # Compute error metrics
        r2_results.append(r2_score(y_test, prediction)) # Compute and store R2
        rmse_results.append(np.sqrt(mean_squared_error(y_test, prediction))) # Compute and store RMSE
        mae_results.append(mean_absolute_error(y_test, prediction)) # Compute and store MAE

    return {
    "regression": "RandomForestRegressor",
    "r2_mean": np.mean(r2_results),
    "r2_std": np.std(r2_results),
    "rmse_mean": np.mean(rmse_results),
    "rmse_std": np.std(rmse_results),
    "mae_mean": np.mean(mae_results),
    "mae_std": np.std(mae_results)
    }    

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

def model_greenhouse_components(greenhouse_components):

    results_table = ["Country", "Linear R2", "Linear RMSE", "Quadratic R2", "Quadratic RMSE"]

    regression_metrics_table = []

    # Select a component from the list of greenhouse gas components to model
    for component in greenhouse_components:

        # Add a row with info of the current component
        regression_metrics_table.append([])
        regression_metrics_table.append([component])
        regression_metrics_table.append(results_table)

        # Select a country from which to model the data
        for country in np_countries:

            # Empty array to save the results for the current country
            country_regression_metrics = [] 
            country_regression_metrics.append(country)

            # Retrieve data for the current country
            df_country = df_copy_pivot.loc[country]

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
            regression_metrics_table.append(country_regression_metrics)

    return regression_metrics_table




# ----------------------------------------------------------------------------------------------------------------------------------
# Dataset preprocessing
# ----------------------------------------------------------------------------------------------------------------------------------

# Create shorter, easier to understand labels for greenhouse gas component names
label_map = {
    "carbon_dioxide_co2_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent": "CO2",
    "hydrofluorocarbons_hfcs_emissions_in_kilotonne_co2_equivalent": "HFCs",
    "methane_ch4_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent": "CH4",
    "nitrogen_trifluoride_nf3_emissions_in_kilotonne_co2_equivalent": "NF3",
    "nitrous_oxide_n2o_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent": "N2O",
    "perfluorocarbons_pfcs_emissions_in_kilotonne_co2_equivalent": "PFCs",
    "sulphur_hexafluoride_sf6_emissions_in_kilotonne_co2_equivalent": "SF6",
    "unspecified_mix_of_hydrofluorocarbons_hfcs_and_perfluorocarbons_pfcs_emissions_in_kilotonne_co2_equivalent": "HFCs/PFCs mix"
}

# Make a cleaner copy of the dataset to work with

df.drop_duplicates(keep="first", inplace=True) # Removes duplicates, keeping the first one, modifies the original df

df_copy = df.copy() # Generate a copy of the dataset to avoid modifying the original (safety measure)
df_copy = df_copy.sort_values(by=["country_or_area", "year"]) # Sort dataset by both country and year
df_copy = df_copy[df_copy["category"].isin(label_map)] # Select relevant components based on label_map
df_copy["total"] = df_copy.groupby("year")["value"].transform("sum") # Total greenhouse gas composition
df_copy["percentage"] = df_copy["value"] / df_copy["total"] * 100 # Percentage per component

# Pivot the copy of the dataset to turn the greenhouse components into columns instead of values
df_copy_pivot = df_copy.pivot(index=["country_or_area","year"], columns="category", values="percentage")
df_copy_pivot = df_copy_pivot.rename(columns=label_map) # Replace label names with shorter versions

# Replace NaN values with 0
df_copy_pivot.fillna(0, inplace=True) # Turn any NaN values to 0

# ----------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis (EDA): Slopes of the greenhouse components' behavior to see if they increase or decrease
# ----------------------------------------------------------------------------------------------------------------------------------

# Denmark data for small scale tests
df_denmark = df_copy_pivot.loc["Denmark"]

# Save the list of countries in the dataset to use as indices for iteration
np_countries = df_copy["country_or_area"].drop_duplicates().to_numpy()

# Indices for the slopes table: country, CO2, NH3, etc.
slopes_table_indices = ["Country"] + df_copy_pivot.columns.to_numpy().tolist()

# Create empty list for the slopes table and add the indices
slopes_table = []
slopes_table.append(slopes_table_indices)

# Iterate over each country's data
for country in np_countries:
    
    # Empty table for saving the slopes of each component per country
    country_component_slopes = []
    country_component_slopes.append(country) # Add name of the country to save it as part of the table to be generated

    # Retrieve data for the current country
    df_country = df_copy_pivot.loc[country]

    # Compute the slopes for each greenhouse gas component in the current country
    for col in df_country.columns:
        m, b = np.polyfit(df_country[col].index.to_numpy(), df_country[col].to_numpy(), 1) # Compute linear slope and intersect
        country_component_slopes.append((m).round(4)) # Round the slop to 4 decimal values and save it

    # Save the current country's result as a row in the slopes table
    slopes_table.append(country_component_slopes)

# Export slope table to a .csv
pd.DataFrame(slopes_table).to_csv(PROJECT_PATH + "01_greenhouse_component_slopes_over_time.csv")

# ----------------------------------------------------------------------------------------------------------------------------------
# Modeling historical data: linear and quadratic regression over the main greenhouse components to model existing data
# ----------------------------------------------------------------------------------------------------------------------------------

# Most relevant components based on their predominance among all countries in the dataset.
greenhouse_components = ["CO2", "HFCs", "CH4", "N2O", "PFCs"]

# Model historical data of the relevant greenhouse gas components
regression_metrics_table = model_greenhouse_components(greenhouse_components)

# Export linear and quadratic error metrics data to a .csv
pd.DataFrame(regression_metrics_table).to_csv(PROJECT_PATH + "02_regression_metrics_table.csv")

# ----------------------------------------------------------------------------------------------------------------------------------
# Predicting future data: Predict the potential behavior of the main greenhouse gas components based on historical data
# ----------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 1: Polynomial and Ridge regressions, grade 1-3
# ---------------------------------------------------------------------------

prediction_results_table = [["Country", "Component", "Best R2 Regression", "Best R2 Value", "Best RMSE Regression", "Best RMSE Value"]]

regressions = {"polynomial","ridge"}
degrees = {1, 2, 3}

for country in np_countries:

    # Retrieve data for the current country
    df_country = df_copy_pivot.loc[country]

    # Predict data for each greenhouse gas component in the country
    for component in greenhouse_components:

        # Retrieve data for the component
        X = df_country.index.to_numpy().reshape(-1,1) # Time (years)
        y = df_country[component].to_numpy().reshape(-1,1) # Greenhouse gas component percentage

        cv_slices = 4
        evaluations = []
        for regression in regressions:
            for degree in degrees:
                # Perform regressions
                evaluations.append(polynomial_tscv(degree, X, y, cv_slices, regression))

        # Select the regression with the best performance (relative to the other regressions)
        # This selection is based on which regression had the largest R2 and smallest RMSE
        best_r2, best_rmse = select_best_fit(evaluations)
        
        # Save the cases where the regression is a good fit (exclude cases with negative R2 values)
        if (best_r2[2] >= 0):
            prediction_results_table.append([country, component, best_r2[1], best_r2[2].round(4), best_rmse[1], best_rmse[2].round(4)])

pd.DataFrame(prediction_results_table).to_csv(PROJECT_PATH + "03_prediction_results_table.csv")

print("Summary of polynomial/ridge regressions:")
print("Total cases: " + str(len(df_copy_pivot)))
print("Cases with good model fit (0<R2<1): " + str(len(prediction_results_table)))
print("Conclusion: Polynomial/ridge regressions are not adequate for this dataset")

# ---------------------------------------------------------------------------
# Test 2: Random Forest Regression and evaluate performance for prediction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Small scale test: Denmark data
# y = CO2 values, X = time (years in the dataset)
y = df_denmark["CO2"].to_numpy() # Greenhouse gas component percentage
X = df_denmark.index.to_numpy().reshape(-1,1) # Time (years)

print("--------------------------------------------------")
print("Random Forest Regressor Test Case:")
print("--------------------------------------------------\n")

rf_results = random_forest_regressor_tscv(X, y, 5)
print_tscv_results("RandomForestRegressor", rf_results)

for country in np_countries:

    # Retrieve data for the current country
    df_country = df_copy_pivot.loc[country]

    # Predict data for each greenhouse gas component in the country
    for component in greenhouse_components:

        # Retrieve data for the component
        X = df_country.index.to_numpy().reshape(-1,1) # Time (years)
        y = df_country[component].to_numpy() # Greenhouse gas component percentage

        cv_slices = 5
        evaluations = []
        # Perform regressions
        evaluations.append(random_forest_regressor_tscv(X, y, cv_slices))

        # Select the regression with the best performance (relative to the other regressions)
        # This selection is based on which regression had the largest R2 and smallest RMSE
        best_r2, best_rmse = select_best_fit(evaluations)
        
        # Save the cases where the regression is a good fit (exclude cases with negative R2 values)
        if (best_r2[2] >= 0):
            prediction_results_table.append([country, component, best_r2[1], best_r2[2].round(4), best_rmse[1], best_rmse[2].round(4)])

pd.DataFrame(prediction_results_table).to_csv(PROJECT_PATH + "04_random_forest_prediction_results_table.csv")
