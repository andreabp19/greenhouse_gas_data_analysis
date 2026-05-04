
# Author: Andrea Pineda
# Date: 20 Apr. 2026
# Summary: data science practice project
# Subject: greenhouse gas dataset from the UN
# Last modified: 4 May. 2026

# ---------------------------------------------------------------------------
# 1. Import libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from config import DATASET_PATH
from config import PROJECT_PATH

# ---------------------------------------------------------------------------
# 2. Import dataset
# ---------------------------------------------------------------------------

# Fix this to get it from the website to avoid writing my directory.
df = pd.read_csv(DATASET_PATH)

# ---------------------------------------------------------------------------
# 3. Define project objectives
# ---------------------------------------------------------------------------

# --------------- 3.2 Summary of the content of the dataset ---------------
# Type and amount of greenhouse emmission per country over a span of several years.

# 2.3 Potential objectives for the project
# A. Relate type of emission per country based on the quantity of the emmission
# B. Relate the amount of different types of emissions over several years per country
# C. Identify the countries with the most emmissions of each emmission type
# D. Predict how much emmissions will a country have based on its previous data

# I like B and D. I will start with B.

# For B:
# The dataset contains 24 years of data for 40+ countries.
# For each country, there is the total yearly amount of emmissions of 10 different greenhouse gas types.

# Break down B further, for starting.
# B.1. For each country, relate the amount of emmissions per type of emmission (how much of each there is when others are present).
# B.2. For each year, identify the top emmissions generated per country
# B.3. Track the behaviour of each emmission across all countries in the dataset over the 24-year span

# Refined idea for B:
# Big (what is it?): Percentage of each emission per country, per year
# Medium (where/when is it?): Identify the trend of each emission type across the 24-year span, per country, then compare globally
# Small (why it matters?): Compare the rates of change (increase/decrease) between the trends in the different types of emissions over time.

# ---------------------------------------------------------------------------
# 4. Clean data
# ---------------------------------------------------------------------------

# Duplicates
df.drop_duplicates(keep="first", inplace=True) # Removes duplicates, keeping the first one, modifies the original df

# Empty or NaN cells
#print(df.isnull().sum()) # Checks if there are null values (either empty cells or NaN), in this case, there are no empty values.

# ---------------------------------------------------------------------------
# 5. EDA (Exploratory Data Analysis)
# ---------------------------------------------------------------------------

# 1. Big (what is it?): Plot the trend of each emission type across time, per country

country_or_area = df["country_or_area"]
year = df["year"]
value = df["value"]
category = df["category"]

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

df_copy = df.copy()
df_copy = df_copy.sort_values(by=["country_or_area", "year"])
df_copy = df_copy[df_copy["category"].isin(label_map)]
df_copy["total"] = df_copy.groupby("year")["value"].transform("sum")
df_copy["percentage"] = df_copy["value"] / df_copy["total"] * 100

# print(df_copy.groupby("year")["percentage"].sum()) # Sanity check, to see if all percentages summ 100

df_copy_pivot = df_copy.pivot(index=["country_or_area","year"], columns="category", values="percentage")
df_copy_pivot = df_copy_pivot.rename(columns=label_map)
df_copy_pivot.fillna(0, inplace=True)
#print(df_copy_pivot)

# Get data from Scandinavian countries: Denmark, Sweden, Norway
df_denmark = df_copy_pivot.loc["Denmark"]
df_sweden = df_copy_pivot.loc["Sweden"]
df_norway = df_copy_pivot.loc["Norway"]

#fig, axes = plt.subplots(nrows=1, ncols=3)

# Plot Denmark Emission Percentages
# df_denmark.plot(ax=axes[0], legend=False)
# axes[0].set_title("Emission Composition Over Time, Denmark 1990-2014")
# axes[0].set_yscale("log")
# axes[0].set_xlabel("Year")
# axes[0].set_ylabel("Percentage (%)")

# Plot Sweden Emission Percentages
# df_sweden.plot(ax=axes[1], legend=False)
# axes[1].set_title("Emission Composition Over Time, Sweden 1990-2014")
# axes[1].set_yscale("log")
# axes[1].set_xlabel("Year")
# axes[1].set_ylabel("Percentage (%)")

# Plot Norway Emission Percentages
# df_norway.plot(ax=axes[2], legend=False)
# axes[2].set_title("Emission Composition Over Time, Norway 1990-2014")
# axes[2].set_yscale("log")
# axes[2].set_xlabel("Year")
# axes[2].set_ylabel("Percentage (%)")

# plt.legend(title="Emission Type", bbox_to_anchor=(0.5,1.165), loc="upper center", ncol=3)
# plt.show()

# 2. Medium (where/when is it?): Compare the emission percentages between the selected countries

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

denmark_describe = df_denmark.describe().T.round(3)
norway_describe = df_norway.describe().T.round(3)
sweden_describe = df_sweden.describe().T.round(3)

denmark_describe.to_csv(PROJECT_PATH + "describe_denmark.csv")
norway_describe.to_csv(PROJECT_PATH + "describe_norway.csv")
sweden_describe.to_csv(PROJECT_PATH + "describe_sweden.csv")

# 3. Small (why is it important?): CO2, N2O, CH4 and HFCs are the highest and HFCs increased over time.
# It would be useful to track these across all countries and find more global patterns of these. 

# Create numpy arrays from the dataframes for each country. First, just the scaninavian countries, then we'll generalize.
# Elements in the array are the components' values

# 3.1 Denmark composition increase slopes
#print("\nDenmark:")
denmark_slopes = []
for col in df_denmark.columns:
    m, b = np.polyfit(df_denmark[col].index.to_numpy(), df_denmark[col].to_numpy(), 1)
    denmark_slopes.append((m).round(4))
    #print(col + ":  " + str((m).round(4)))

# 3.2 Norway composition increase slopes
#print("\nNorway:")
norway_slopes = []
for col in df_norway.columns:
    m, b = np.polyfit(df_norway[col].index.to_numpy(), df_norway[col].to_numpy(), 1)
    norway_slopes.append((m).round(4))
    #print(col + ":  " + str((m).round(4)))

# 3.3 Sweden composition increase slopes
#print("\nSweden:")
sweden_slopes = []
for col in df_sweden.columns:
    m, b = np.polyfit(df_sweden[col].index.to_numpy(), df_sweden[col].to_numpy(), 1)
    sweden_slopes.append((m).round(4))
    #print(col + ":  " + str((m).round(4)))

pd.DataFrame(denmark_slopes).to_csv(PROJECT_PATH + "slopes_denmark.csv")
pd.DataFrame(norway_slopes).to_csv(PROJECT_PATH + "slopes_norway.csv")
pd.DataFrame(sweden_slopes).to_csv(PROJECT_PATH + "slopes_sweden.csv")

# ---------------------------------------------------------------------------
# 4. Model data
# ---------------------------------------------------------------------------

# 4.1 Apply a linear or quadratic regression to the most relevant components in each country's data (historical modeling)

# Generalize per component, for each country.
# Generate a .csv with the linear and quadratic R2 and RMSE per greenhouse gas component for all countries.
# Total: 5 .csv
# Components to model: CO2, HFCs, CH4, N2O, PFCs, the others are too scarse in the dataset so as to model them.

# Model the linear and quadratic regression of a single greenhouse component in a single country (small scale test)

y = df_denmark["CO2"].to_numpy().reshape(-1,1)
X = df_denmark.index.to_numpy().reshape(-1,1)

# Linear Model
lin_model = LinearRegression().fit(X, y)
y_lin = lin_model.predict(X)


# Quadratic Model

poly2 = PolynomialFeatures(degree=2)
X_quad = poly2.fit_transform(X)

quad_model = LinearRegression().fit(X_quad, y)
y_quad = quad_model.predict(X_quad)

# Plot regression results

plt.figure(figsize=(10,6))

plt.plot(X, y, label="CO2 percentages")
plt.plot(X, y_lin, label="Linear prediction")
plt.plot(X, y_quad, label="Quadratic prediction")
plt.title("Denmark CO2 with Linear and Quadratic Regression")
plt.legend()

plt.show()

# Calculate error metrics: R2, mean square

print("Linear R2: ", r2_score(y, y_lin))
print("Quadratic R2: ", r2_score(y, y_quad))

print("Linear RMSE: ", np.sqrt(mean_squared_error(y, y_lin)))
print("Quadratic RMSE: ", np.sqrt(mean_squared_error(y, y_quad)))