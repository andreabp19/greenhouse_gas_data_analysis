
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
import matplotlib.pyplot as plt

# Make a config.py and save the dataset's and project's path in: DATASET_PATH, PROJECT_PATH
from config import PROJECT_PATH # From local file with the path to the project's folder

# Custom functions and variables in functions.py and preprocessing.py
from functions import apply_regressions, apply_regressions_tscv
from preprocessing import df_workset, countries_in_dataset, components_in_dataset, components_to_model

# ----------------------------------------------------------------------------------------------------------------------------------
# Constants, variables and arrays
# ----------------------------------------------------------------------------------------------------------------------------------

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
# Modeling historical data: linear and quadratic regression over the main greenhouse components to model existing data
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

#--------------- Selecting and counting best fit for modeling the greenhouse gas component data ------------------------------------

print("- Creating summary of best fit per component:")

df_historical = pd.DataFrame(modeling_raw_results) # Generate a datafram out of the regression results table
df_historical.columns = df_historical.iloc[0] # Set the row with labels as column names
df_historical = df_historical[1:].reset_index(drop=True) # Remove the row that has the would-be column labels

df_historical_r2 = df_historical.drop(columns=df_historical.filter(regex=r"RMSE$").columns) # Save a copy with only the R2 results

# Remove seemingly perfect scores (R2 = 1), as those are probably bugged and highly improbable (and could impact analysis)
df_historical_r2 = df_historical_r2[~(df_historical_r2[df_historical_r2.columns[2:-2]] == 1.0).any(axis=1)]

# Identify best regression fit and save it in new columns
df_historical_r2["Best Fit"] = df_historical_r2.iloc[:, 2:].idxmax(axis=1) # Label of the best regression fit
df_historical_r2["Best R2"] = df_historical_r2.iloc[:, 2:-1].max(axis=1) # R2 score of the best regression

# Remove all values below the acceptable tolerance for R2 scores
df_historical_r2 = df_historical_r2[(df_historical_r2["Best R2"] >= R2_TOLERANCE)]

df_best_modeling_fits = df_historical_r2.groupby("Component")["Best Fit"].value_counts() # Count best fits per component
df_best_modeling_fits = pd.DataFrame(df_best_modeling_fits).reset_index() # Convert from Series to Dataframe
df_best_modeling_fits["Best Fit"] = df_best_modeling_fits["Best Fit"].str.replace(r' R2', '', regex=True) # Remove " R2" from labels
df_best_modeling_fits = df_best_modeling_fits.pivot(index="Component", columns="Best Fit", values="count").fillna(0)

print("- Finished summary of best fit per component:")

# ----------------------------------------------------------------------------------------------------------------------------------
# Predicting future data: Predict the potential behavior of the main greenhouse gas components based on historical data
# ----------------------------------------------------------------------------------------------------------------------------------

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

#--------------- Prediction: Identify acceptable fit cases -------------------------------------------------------------------------

df_prediction = pd.DataFrame(prediction_raw_results) # Generate DataFrame with the prediction error metrics table
df_prediction.columns = df_prediction.iloc[0] # Set the row with labels as column names
df_prediction = df_prediction[1:].reset_index(drop=True) # Remove the row that has the would-be column labels

# Remove seemingly perfect scores (R2 = 1), as those are probably bugged and highly improbable (and could impact analysis)
df_prediction = df_prediction[~(df_prediction[df_prediction.columns[2:]] == 1.0).any(axis=1)]

df_training = df_prediction.drop(columns=df_prediction.filter(regex=r"Pred").columns) # Make a copy with the training results
df_prediction = df_prediction.drop(columns=df_prediction.filter(regex=r"Train").columns) # Make a copy with the prediction results

# Identify best regression fit and save it in new columns
df_training["Best Fit"] = df_training.iloc[:, 2:].idxmax(axis=1) # Label of the best regression fit
df_training["Best R2"] = df_training.iloc[:, 2:-1].max(axis=1) # R2 score of the best regression
df_prediction["Best Fit"] = df_prediction.iloc[:, 2:].idxmax(axis=1) # Label of the best regression fit
df_prediction["Best R2"] = df_prediction.iloc[:, 2:-1].max(axis=1) # R2 score of the best regression
#df_best_prediction_fits = df_best_prediction_fits.pivot(index="Component", columns="Best Fit", values="Best R2").fillna(0)

print(df_training)
print(df_prediction)

df_training = pd.DataFrame(df_training).reset_index(drop=True) # Convert from Series to Dataframe
df_training["Best Fit"] = df_training["Best Fit"].str.replace(r' R2 Mean', '', regex=True) # Remove " R2" from labels
df_prediction = pd.DataFrame(df_prediction).reset_index(drop=True) # Convert from Series to Dataframe
df_prediction["Best Fit"] = df_prediction["Best Fit"].str.replace(r' R2 Mean', '', regex=True) # Remove " R2" from labels

# Remove all values below the acceptable tolerance for R2 scores
df_training = df_training[(df_training["Best R2"] >= R2_TOLERANCE)]
df_prediction = df_prediction[(df_prediction["Best R2"] >= R2_TOLERANCE)]

print(df_training)
print(df_prediction)

df_training.drop(columns="Country") # Remove Country column
df_training = df_training.groupby("Component")["Best Fit"].value_counts().reset_index(name="count") # Count best fits per component
df_training = df_training.pivot(index="Component", columns="Best Fit", values="count").fillna(0)
df_prediction.drop(columns="Country") # Remove Country column
df_prediction = df_prediction.groupby("Component")["Best Fit"].value_counts().reset_index(name="count") # Count best fits per component
df_prediction = df_prediction.pivot(index="Component", columns="Best Fit", values="count").fillna(0)

print(df_training)
print(df_prediction)

# ----------------------------------------------------------------------------------------------------------------------------------
# Plotting area (all plots generated here after data is processed)
# ----------------------------------------------------------------------------------------------------------------------------------

#--------------- Modeling best-fit regression summary for historical data modeling -------------------------------------------------

# Create plot
colors = ["#B7D3C2", "#F7A9A8", "#B8C0FF"]
ax = df_best_modeling_fits.plot(kind='bar', stacked=True, figsize=(12,8), color=colors)

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
plt.title("Number of best-fit cases per regression type for modeling the available greenhouse gas component data")
plt.xlabel("Greenhouse Gas Component")
plt.ylabel("Number of best fit cases per regression type")
plt.show()

#--------------- Prediction best fit regressions during training -------------------------------------------------------------------

# Create plot
colors = ["#B7D3C2", "#F7A9A8", "#B8C0FF"]
ax = df_training.plot(kind='bar', stacked=True, figsize=(12,8), color=colors)

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
plt.title("Number of best-fit cases per regression type during training")
plt.xlabel("Greenhouse Gas Component")
plt.ylabel("Number of best fit cases per regression type")
plt.show()

#--------------- Prediction best fit regressions during prediction -----------------------------------------------------------------

# Create plot
colors = ["#B7D3C2", "#F7A9A8", "#B8C0FF"]
ax = df_prediction.plot(kind='bar', stacked=True, figsize=(12,8), color=colors)

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
plt.title("Number of best-fit cases per regression type during prediction")
plt.xlabel("Greenhouse Gas Component")
plt.ylabel("Number of best fit cases per regression type")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------
# Export results to .csv files
# ----------------------------------------------------------------------------------------------------------------------------------

# Export slope results table
pd.DataFrame(slopes_table).to_csv(PROJECT_PATH + "01_greenhouse_component_slopes_over_time.csv")

# Export historical modeling error metrics table
pd.DataFrame(modeling_raw_results).to_csv(PROJECT_PATH + "02_modeling_raw_results.csv")

# Export the summary of best regression fits per greenhouse gas component
df_best_modeling_fits.to_csv(PROJECT_PATH + "03_best_fit_per_component.csv")

# Export prediction error metrics table
pd.DataFrame(prediction_raw_results).to_csv(PROJECT_PATH + "04_prediction_raw_results.csv")