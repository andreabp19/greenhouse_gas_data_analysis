
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: Coursera data science project
# Subject: greenhouse gas dataset from the UN (obtained from Kaggle)
# Last modified: 4 May. 2026

# ----------------------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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

def polynomial_tscv(pol_degree, X, y, nsplits):

    r2_results = []
    rmse_results = []
    mae_results = []

    X_centered = X - X.min() # Center years around 0 instead of 1000+ dates
    X_set = PolynomialFeatures(pol_degree).fit_transform(X_centered) # Polynomial features

    # Generate cross validation splits
    tscv = TimeSeriesSplit(n_splits=nsplits)
    splits = list(tscv.split(X_centered,y))
    
    for train_idx, test_idx in splits:
        
        # Slice training and test subsets
        X_train, X_test = X_set[train_idx,:], X_set[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        # Model train and predict
        train = LinearRegression().fit(X_train, y_train)
        prediction = train.predict(X_test)

        # Compute error metrics
        r2_results.append(r2_score(y_test, prediction)) # Compute and store R2
        rmse_results.append(np.sqrt(mean_squared_error(y_test, prediction))) # Compute and store RMSE
        mae_results.append(mean_absolute_error(y_test, prediction)) # Compute and store MAE

    return {
    "r2_mean": np.mean(r2_results),
    "r2_std": np.std(r2_results),
    "rmse_mean": np.mean(rmse_results),
    "rmse_std": np.std(rmse_results),
    "mae_mean": np.mean(mae_results),
    "mae_std": np.std(mae_results)
    }

def print_polynomial_tscv_results(text, results):
    print("\n--------------------------------------------------")
    print(text + " Error Metrics:")
    print("--------------------------------------------------")
    print("R2 mean: " + str(results["r2_mean"]))
    print("R2 std.dev: " + str(results["r2_std"]))
    print("RMSE mean: " + str(results["rmse_mean"]))
    print("RMSE std.dev: " + str(results["rmse_std"]))
    print("MAE mean: " + str(results["mae_mean"]))
    print("MAE std.dev: " + str(results["mae_std"]))

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

# Components to model: CO2, HFCs, CH4, N2O, PFCs, the others are too scarse in the dataset so as to model them.

regression_metrics_table_indices = ["Country", "Linear R2", "Linear RMSE", "Quadratic R2", "Quadratic RMSE"]
greenhouse_components_to_model = ["CO2", "HFCs", "CH4", "N2O", "PFCs"]

regression_metrics_table = []

# Select a component from the list of greenhouse gas components to model
for component in greenhouse_components_to_model:

    # Add a row with info of the current component
    regression_metrics_table.append([])
    regression_metrics_table.append([component])
    regression_metrics_table.append(regression_metrics_table_indices)

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

# Export linear and quadratic error metrics data to a .csv
pd.DataFrame(regression_metrics_table).to_csv(PROJECT_PATH + "02_regression_metrics_table.csv")


# ---------------------------------------------------------------------------
# Small scale test for linear and quadratic regression: Denmark data

# y = CO2 values, X = time (years in the dataset)
y = df_denmark["CO2"].to_numpy().reshape(-1,1) # Greenhouse gas component percentage
X = df_denmark.index.to_numpy().reshape(-1,1) # Time (years)

# Linear Model
lin_model = LinearRegression().fit(X, y)
y_lin = lin_model.predict(X)

# Quadratic Model
poly2 = PolynomialFeatures(degree=2)
X_quad = poly2.fit_transform(X)
quad_model = LinearRegression().fit(X_quad, y)
y_quad = quad_model.predict(X_quad)

# ----------------------------------------------------------------------------------------------------------------------------------
# Predicting future data: Predict the potential behavior of the main greenhouse gas components based on historical data
# ----------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 1: Polynomial Regression (Grades: 1, 2 and 3) on Denmark data
# ---------------------------------------------------------------------------

# Arrays for error metrics
lin_r2 = []
lin_rmse = []
quad_r2 = []
quad_rmse = []
cube_r2 = []
cube_rmse = []

results = polynomial_tscv(1, X, y, 5)
print_polynomial_tscv_results("Linear", results)

results = polynomial_tscv(2, X, y, 5)
print_polynomial_tscv_results("Quadratic", results)

results = polynomial_tscv(3, X, y, 5)
print_polynomial_tscv_results("Cubic", results)