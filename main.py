
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: Coursera data science project
# Subject: greenhouse gas dataset from the UN (obtained from Kaggle)
# Dataset: https://www.kaggle.com/datasets/unitednations/international-greenhouse-gas-emissions/data
# Last modified: 18 May. 2026

# ----------------------------------------------------------------------------------------------------------------------------------
# Import libraries, functions and variables
# ----------------------------------------------------------------------------------------------------------------------------------

# Python Libraries
import pandas as pd
import numpy as np

# Make a config.py and save the dataset's and project's path in: DATASET_PATH, PROJECT_PATH
from config import PROJECT_PATH # From local file with the path to the project's folder

# Custom functions and variables in functions.py and preprocessing.py
from functions import apply_regressions, apply_regressions_tscv, plot_regression_count
from functions import df_from_raw, get_best_regression_fit
from preprocessing import df_workset, countries_in_dataset, components_in_dataset, components_to_model

# ----------------------------------------------------------------------------------------------------------------------------------
# Constants, variables and arrays
# ----------------------------------------------------------------------------------------------------------------------------------

# Colors
colors = ["#B7D3C2", "#F7A9A8", "#B8C0FF"]

# Constants
R2_TOLERANCE = 0.6 # 0.6 <= R2 < 1 acceptable-to-good fit

# Trend slopes table: for saving and exporting the results of slope computation for increase/decrease of greenhouse components
slopes_table = []
slopes_table.append(["Country"] + components_in_dataset) # Table column names

# Regression configs
regressions = ["polynomial","ridge","random_forest"] # Regressions to use
degrees = [1, 2, 3] # Degrees for Polynomial and Ridge regressions
cv_slices = 2 # Dataset slices for Time-Series Cross Validation

# Table results (with their column names)
modeling_raw_results = [
    ["Country", "Component", "Pol1 R2", "Pol1 RMSE", "Pol2 R2",
     "Pol2 RMSE", "Pol3 R2", "Pol3 RMSE", "Ridge1 R2", "Ridge1 RMSE",
     "Ridge2 R2", "Ridge2 RMSE", "Ridge3 R2", "Ridge3 RMSE", "RandomForest R2", "RandomForest RMSE"]]

prediction_raw_results = [
    ["Country", "Component",
     "Polynomial1 Train R2", "Polynomial1 Pred R2",
     "Polynomial2 Train R2", "Polynomial2 Pred R2",
     "Polynomial3 Train R2", "Polynomial3 Pred R2",
     "Ridge1 Train R2", "Ridge1 Pred R2",
     "Ridge2 Train R2", "Ridge2 Pred R2",
     "Ridge3 Train R2", "Ridge3 Pred R2",
     "RandomForest Train R2", "RandomForest Pred R2"]]

# ----------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis (EDA): Slopes of the greenhouse components' behavior to see their increase or decrease over time
# ----------------------------------------------------------------------------------------------------------------------------------

print("- Starting slope computing:")
# Iterate over each country's data
for country in countries_in_dataset:
    
    # Empty table for saving the slopes of each component per country
    country_component_slopes = []
    country_component_slopes.append(country) # Add name of the country to save it as part of the table to be generated

    # Retrieve data for the current country
    df_country = df_workset.loc[country]

    # Compute the slopes for each greenhouse gas component in the current country
    for col in df_country.columns:
        m, b = np.polyfit(df_country[col].index.to_numpy(), df_country[col].to_numpy(), 1) # Compute linear slope and intersect
        country_component_slopes.append((m).round(4)) # Round the slope to 4 decimal values and save it

    print("    - " + country) # Print for debugging

    # Save the current country's result as a row in the slopes table
    slopes_table.append(country_component_slopes)

print("- Finished slope computing")

# ----------------------------------------------------------------------------------------------------------------------------------
# Regression analysis: polynomial, ridge and random forest regressions for historical data modeling and future data prediction
# ----------------------------------------------------------------------------------------------------------------------------------

#--------------- Modeling: Polynomial, Ridge and Random Forest Regressions, with error metrics -------------------------------------

print("- Starting raw historical data modeling:")
for country in countries_in_dataset:

    df_country = df_workset.loc[country] # Retrieve data for the current country

    # Model the different components in the list with the chosen variations
    for component in components_to_model:
        regression_metrics = apply_regressions(regressions, degrees, df_country, component)
        modeling_raw_results.append([country, component] + regression_metrics)
        
        print("    - " + country + ", " + component) # Print for debugging

print("- Finished raw historical data modeling")

#--------------- Prediction: Polynomial, Ridge and Random Forest Regressions with Time-Series Cross Validation ---------------------

print("- Starting data prediction:")
for country in countries_in_dataset:

    df_country = df_workset.loc[country] # Retrieve data for the current country

    # Predict the different components in the list with the chosen regressions
    for component in components_to_model:
        regression_metrics = apply_regressions_tscv(regressions, degrees, cv_slices, df_country, component)
        prediction_raw_results.append([country, component] + regression_metrics)

        print("    - " + country + ", " + component) # Print for debugging

print("- Finished data prediction")

# ----------------------------------------------------------------------------------------------------------------------------------
# Postprocesing: Counting best regression fit cases with each applied regression for modeling and prediction
# ----------------------------------------------------------------------------------------------------------------------------------

print("- Creating summary of best fit per component:")

# 1. Create dataframe from raw results
df_historical = df_from_raw(modeling_raw_results)
df_prediction = df_from_raw(prediction_raw_results)

# 2. Remove unnecessary columns in the dataframe, keep only R2 score results
df_modeling = df_historical.drop(columns=df_historical.filter(regex=r"RMSE$").columns) # Save a copy with only the R2 results

# 3. Remove seemingly perfect scores (R2 = 1), as those are probably bugged and could impact interpretation
df_modeling = df_modeling[~(df_modeling[df_modeling.columns[2:-2]] == 1.0).any(axis=1)]
df_prediction = df_prediction[~(df_prediction[df_prediction.columns[2:]] == 1.0).any(axis=1)]

# 4. Separate training and prediction dataframe results
df_training = df_prediction.drop(columns=df_prediction.filter(regex=r"Pred").columns) # Make a copy with the training results
df_prediction = df_prediction.drop(columns=df_prediction.filter(regex=r"Train").columns) # Make a copy with the prediction results

# 5. Add new columns with best regression results: regression label, r2_score
df_modeling = get_best_regression_fit(df_modeling)
df_training = get_best_regression_fit(df_training)
df_prediction = get_best_regression_fit(df_prediction)

# 6. Filter out every column except the best fit data, and remove extra characters from regression names
df_modeling["Best Fit"] = df_modeling["Best Fit"].str.replace(r' R2', '', regex=True) # Remove " R2" from labels
df_training["Best Fit"] = df_training["Best Fit"].str.replace(r' R2', '', regex=True) # Remove " R2" from labels
df_prediction["Best Fit"] = df_prediction["Best Fit"].str.replace(r' R2', '', regex=True) # Remove " R2" from labels

# 7. Filter out all results below the acceptable R2 score (0.6 <= R2 < 1)
df_modeling = df_modeling[(df_modeling["Best R2"] >= R2_TOLERANCE)]
df_training = df_training[(df_training["Best R2"] >= R2_TOLERANCE)]
df_prediction = df_prediction[(df_prediction["Best R2"] >= R2_TOLERANCE)]

# 8. Remove "Country" column in training and prediction dataframes
df_training.drop(columns="Country")
df_prediction.drop(columns="Country")

# 9. Organize data inside each dataframe and count total cases per regression in "Best Fit"
df_modeling = df_modeling.groupby("Component")["Best Fit"].value_counts().reset_index() 
df_training = df_training.groupby("Component")["Best Fit"].value_counts().reset_index(name="count") 
df_prediction = df_prediction.groupby("Component")["Best Fit"].value_counts().reset_index(name="count")

# 10. Pivot tables so regression labels are the columns and total count are the values in the cells (and replace NaNs with 0s)
df_modeling = df_modeling.pivot(index="Component", columns="Best Fit", values="count").fillna(0)
df_training = df_training.pivot(index="Component", columns="Best Fit", values="count").fillna(0)
df_prediction = df_prediction.pivot(index="Component", columns="Best Fit", values="count").fillna(0)

print(df_modeling)
print(df_training)
print(df_prediction)

print("- Finished summary of best fit per component:")

# ----------------------------------------------------------------------------------------------------------------------------------
# Plotting area
# ----------------------------------------------------------------------------------------------------------------------------------

plot_regression_count(colors, df_modeling, "Number of best-fit cases per regression type for modeling the available greenhouse gas component data")
plot_regression_count(colors, df_training, "Number of best-fit cases per regression type during training")
plot_regression_count(colors, df_prediction, "Number of best-fit cases per regression type during prediction")

# ----------------------------------------------------------------------------------------------------------------------------------
# Export results to .csv files
# ----------------------------------------------------------------------------------------------------------------------------------

# Export raw result tables
pd.DataFrame(slopes_table).to_csv(PROJECT_PATH + "01_greenhouse_component_slopes_over_time.csv")
pd.DataFrame(modeling_raw_results).to_csv(PROJECT_PATH + "02_modeling_raw_results.csv")
pd.DataFrame(prediction_raw_results).to_csv(PROJECT_PATH + "03_prediction_raw_results.csv")

# Export dataframes
df_modeling.to_csv(PROJECT_PATH + "04_best_regression_fits_modeling.csv")
df_training.to_csv(PROJECT_PATH + "05_best_regression_fits_training.csv")
df_prediction.to_csv(PROJECT_PATH + "06_best_regression_fits_prediction.csv")