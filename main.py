
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: Coursera data science project
# Subject: greenhouse gas dataset from the UN (obtained from Kaggle)
# Dataset: https://www.kaggle.com/datasets/unitednations/international-greenhouse-gas-emissions/data
# Last modified: 11 May. 2026

# ----------------------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For using the code, make a config.py and save the dataset's and project's path in: DATASET_PATH, PROJECT_PATH
from config import DATASET_PATH # From local file with the path to the dataset
from config import PROJECT_PATH # From local file with the path to the project's folder

# Importing custom functions from functions.py
from functions import regression_predict_tscv, print_tscv_results, select_best_fit, regression_model

# ----------------------------------------------------------------------------------------------------------------------------------
# Import dataset
# ----------------------------------------------------------------------------------------------------------------------------------

# Read dataset from the .csv
df = pd.read_csv(DATASET_PATH)

# Configs for printing the data in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

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
# Arrays for saving results or important data to be used later
# ----------------------------------------------------------------------------------------------------------------------------------

# 1. List of countries in the dataset: for iteration in modeling and prediction
countries_in_dataset = df_copy["country_or_area"].drop_duplicates().to_numpy()

# 2. Denmark data: small test case for trying new modeling before applying to all countries in the dataset
df_denmark = df_copy_pivot.loc["Denmark"]

# 3. List of greenhouse gas components in the dataset and subset to be used in prediction
components_in_dataset = df_copy_pivot.columns.to_numpy().tolist()
components_to_predict = ["CO2", "HFCs", "CH4", "N2O", "PFCs"]

# 4. Trend slopes table: for saving and exporting the results of slope computation for increase/decrease of greenhouse components
slopes_table = []
slopes_table.append(["Country"] + components_in_dataset) # Table column names

# ----------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis (EDA): Slopes of the greenhouse components' behavior to see their increase or decrease over time
# ----------------------------------------------------------------------------------------------------------------------------------

# Iterate over each country's data
for country in countries_in_dataset:
    
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

# ----------------------------------------------------------------------------------------------------------------------------------
# Modeling historical data: linear and quadratic regression over the main greenhouse components to model existing data
# ----------------------------------------------------------------------------------------------------------------------------------

historical_modeling_table = regression_model(components_to_predict, countries_in_dataset, df_copy_pivot)

# ----------------------------------------------------------------------------------------------------------------------------------
# Predicting future data: Predict the potential behavior of the main greenhouse gas components based on historical data
# ----------------------------------------------------------------------------------------------------------------------------------

#--------------- Polynomial, Ridge and Random Forest Regression and error metrics --------------------------------------------------

prediction_results_table = [["Country", "Component", "Best R2 Regression", "Best R2 Value", "Best RMSE Regression", "Best RMSE Value"]]

regressions = {"polynomial","ridge","random_forest"}
degrees = {1, 2, 3}

for country in countries_in_dataset:

    # Retrieve data for the current country
    df_country = df_copy_pivot.loc[country]

    # Predict data for each greenhouse gas component in the country
    for component in components_to_predict:

        # Retrieve data for the component
        X = df_country.index.to_numpy().reshape(-1,1) # Time (years)
        
        cv_slices = 4
        evaluations = []

        for regression in regressions:

            if(regression=="polynomial" or regression=="ridge"):
                y = df_country[component].to_numpy().reshape(-1,1) # Greenhouse gas component percentage

                for degree in degrees:
                    # Perform regressions
                    evaluations.append(regression_predict_tscv(degree, X, y, cv_slices, regression))

            elif(regression=="random_forest"):
                y = df_country[component].to_numpy() # Greenhouse gas component percentage
                evaluations.append(regression_predict_tscv(1, X, y, cv_slices, regression))

        # Select the regression with the best performance (relative to the other regressions)
        # This selection is based on which regression had the largest R2 and smallest RMSE
        best_r2, best_rmse = select_best_fit(evaluations)
        
        # Save the cases where the regression is a good fit (exclude cases with negative R2 values)
        if (best_r2[2] >= 0):
            prediction_results_table.append([country, component, best_r2[1], best_r2[2].round(4), best_rmse[1], best_rmse[2].round(4)])


# ----------------------------------------------------------------------------------------------------------------------------------
# Export results to .csv files
# ----------------------------------------------------------------------------------------------------------------------------------

# Export slope results table
pd.DataFrame(slopes_table).to_csv(PROJECT_PATH + "01_greenhouse_component_slopes_over_time.csv")

# Export historical modeling error metrics table
pd.DataFrame(historical_modeling_table).to_csv(PROJECT_PATH + "02_historical_modeling_table.csv")

# Export prediction error metrics table
pd.DataFrame(prediction_results_table).to_csv(PROJECT_PATH + "03_prediction_results_table.csv")