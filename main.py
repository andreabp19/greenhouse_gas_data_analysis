
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: Coursera data science project
# Subject: greenhouse gas dataset from the UN (obtained from Kaggle)
# Dataset: https://www.kaggle.com/datasets/unitednations/international-greenhouse-gas-emissions/data
# Last modified: 15 May. 2026

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
from functions import regression_predict_tscv, print_tscv_results, select_best_prediction_fit, regression_model
from preprocessing import df_workset, countries_in_dataset, components_in_dataset, components_to_predict

# ----------------------------------------------------------------------------------------------------------------------------------
# Arrays for relevant data
# ----------------------------------------------------------------------------------------------------------------------------------

# Trend slopes table: for saving and exporting the results of slope computation for increase/decrease of greenhouse components
slopes_table = []
slopes_table.append(["Country"] + components_in_dataset) # Table column names

# Regressions and degrees to be used for modeling and prediction
regressions = ["polynomial","ridge","random_forest"]
degrees = [1, 2, 3]

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

modeling_raw_results = [
    ["Country", "Component", "Pol1 R2", "Pol1 RMSE", "Pol2 R2",
     "Pol2 RMSE", "Pol3 R2", "Pol3 RMSE", "Ridge1 R2", "Ridge1 RMSE",
     "Ridge2 R2", "Ridge2 RMSE", "Ridge3 R2", "Ridge3 RMSE", "RandomForest R2", "RandomForest RMSE"]]

print("- Starting raw historical data modeling:")
metrics_dict = []
for country in countries_in_dataset:

    # Retrieve data for the current country
    df_country = df_workset.loc[country]

    for component in components_to_predict:
        regresion_modeling_results = []

        for regression in regressions:
            if(regression=="polynomial" or regression=="ridge"):
                for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                    metrics = regression_model(degree, regression, df_country, component)
                    regresion_modeling_results.extend(np.round(metrics[1:3],4))

            elif(regression=="random_forest"):
                metrics = regression_model(1, regression, df_country, component)
                regresion_modeling_results.extend(np.round(metrics[1:3],4))

        print("    - " + country + ", " + component) # Print for debugging

        modeling_raw_results.append([country, component] + regresion_modeling_results,)

print("- Finished raw historical data modeling")

print("- Creating summary of best fit per component:")

df_historical = pd.DataFrame(modeling_raw_results) # Generate a datafram out of the regression results table
df_historical.columns = df_historical.iloc[0] # Set the row with labels as column names
df_historical = df_historical[1:].reset_index(drop=True) # Remove the row that has the would-be column labels

df_historical_r2 = df_historical.drop(columns=df_historical.filter(regex=r"RMSE$").columns) # Save a copy without the RMSE columns

# Remove seemingly perfect scores (R2 = 1), as those are probably bugged and highly improbable (and could impact analysis)
df_historical_r2 = df_historical_r2[~(df_historical_r2[df_historical_r2.columns[2:-2]] == 1.0).any(axis=1)]

# Create new columns with the best fit data
df_historical_r2["Best Fit"] = df_historical_r2.iloc[:, 2:].idxmax(axis=1) # Label of the best regression fit
df_historical_r2["Best R2"] = df_historical_r2.iloc[:, 2:-1].max(axis=1) # R2 score of the best regression
df_best_fit_per_component = df_historical_r2.groupby("Component")["Best Fit"].value_counts() # Count best fits per component

print("- Finished summary of best fit per component:")

# ----------------------------------------------------------------------------------------------------------------------------------
# Predicting future data: Predict the potential behavior of the main greenhouse gas components based on historical data
# ----------------------------------------------------------------------------------------------------------------------------------

#--------------- Polynomial, Ridge and Random Forest Regression and error metrics --------------------------------------------------

prediction_results_table = [["Country", "Component", "Best R2 Regression", "Best R2 Value", "Best RMSE Regression", "Best RMSE Value"]]

print("- Starting data prediction:")
for country in countries_in_dataset:

    # Retrieve data for the current country
    df_country = df_workset.loc[country]

    # Predict data for each greenhouse gas component in the country
    for component in components_to_predict:

        cv_slices = 4
        evaluations = []

        for regression in regressions:

            if(regression=="polynomial" or regression=="ridge"):
                for degree in degrees: # Evaluate the polynomial or ridge regressions for degrees 1-3
                    evaluations.append(regression_predict_tscv(degree, cv_slices, regression, df_country, component))

            elif(regression=="random_forest"):
                evaluations.append(regression_predict_tscv(1, cv_slices, regression, df_country, component))

        # Select the regression with the best performance (relative to the other regressions)
        # This selection is based on which regression had the largest R2 and smallest RMSE
        best_r2, best_rmse = select_best_prediction_fit(evaluations)

        print("    - " + country + ", " + component) # Print for debugging

        # Save the cases where the regression is a good fit (exclude cases with negative R2 values)
        if (best_r2[2] >= 0):
            prediction_results_table.append([country, component, best_r2[1], best_r2[2].round(4), best_rmse[1], best_rmse[2].round(4)])

print("- Finished data prediction")


# ----------------------------------------------------------------------------------------------------------------------------------
# Plotting area (all plots generated here after data is processed)
# ----------------------------------------------------------------------------------------------------------------------------------

#--------------- Modeling best-fit regression summary for historical data modeling -------------------------------------------------

print("\n\n\n")

df_best_fit_per_component = pd.DataFrame(df_best_fit_per_component).reset_index() # Convert from Series to Dataframe
df_best_fit_per_component["Best Fit"] = df_best_fit_per_component["Best Fit"].str.replace(r' R2', '', regex=True) # Remove " R2" from labels
df_best_fit_per_component = df_best_fit_per_component.pivot(index="Component", columns="Best Fit", values="count").fillna(0)

# Create plot
colors = ["#B7D3C2", "#F7A9A8", "#B8C0FF"]
ax = df_best_fit_per_component.plot(kind='bar', stacked=True, figsize=(12,8), color=colors)

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

print("\n\n\n")

# ----------------------------------------------------------------------------------------------------------------------------------
# Export results to .csv files
# ----------------------------------------------------------------------------------------------------------------------------------

# Export slope results table
pd.DataFrame(slopes_table).to_csv(PROJECT_PATH + "01_greenhouse_component_slopes_over_time.csv")

# Export historical modeling error metrics table
pd.DataFrame(modeling_raw_results).to_csv(PROJECT_PATH + "02_modeling_raw_results.csv")

# Export the summary of best regression fits per greenhouse gas component
df_best_fit_per_component.to_csv(PROJECT_PATH + "03_best_fit_per_component.csv")

# Export prediction error metrics table
pd.DataFrame(prediction_results_table).to_csv(PROJECT_PATH + "04_prediction_results_table.csv")