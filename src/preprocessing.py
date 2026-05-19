
# Author: Andrea Pineda
# Date: 11 May. 2026
# Summary: Dataset cleaning and preprocessing for routines used in main.py
# Last modified: 11 May. 2026

# ----------------------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For using the code, make a config.py and save the dataset's and project's path in: DATASET_PATH, PROJECT_PATH
from config import DATASET_PATH # From local file with the path to the dataset

# ----------------------------------------------------------------------------------------------------------------------------------
# Import dataset
# ----------------------------------------------------------------------------------------------------------------------------------

# Read dataset from the .csv
df = pd.read_csv(DATASET_PATH)

# ----------------------------------------------------------------------------------------------------------------------------------
# Terminal config
# ----------------------------------------------------------------------------------------------------------------------------------

# Remove limit of characters for printing full tables in the terminal
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
df_workset = df_copy.pivot(index=["country_or_area","year"], columns="category", values="percentage")
df_workset = df_workset.rename(columns=label_map) # Replace label names with shorter versions

# Replace NaN values with 0
df_workset.fillna(0, inplace=True) # Turn any NaN values to 0

# ----------------------------------------------------------------------------------------------------------------------------------
# Arrays for relevant subsets, indices and columns
# ----------------------------------------------------------------------------------------------------------------------------------

# 1. List of countries in the dataset: for iteration in modeling and prediction
countries_in_dataset = df_copy["country_or_area"].drop_duplicates().to_numpy()

# 2. Denmark data: small test case for trying new modeling before applying to all countries in the dataset
df_denmark = df_workset.loc["Denmark"]

# 3. List of greenhouse gas components in the dataset and subset to be used in prediction
components_in_dataset = df_workset.columns.to_numpy().tolist()
components_to_model = ["CO2", "HFCs", "CH4", "N2O", "PFCs"]
