# ZOMATO RESTAURANT DATA CLEANER + EDA

                                    # STEP 1: Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

                                        # STEP 2: Load CSV

# Read the Zomato CSV file into a DataFrame
df = pd.read_csv("zomato.csv")
# Display first 5 rows
print(df.head())

                                # STEP 3: Understand the Dataset

# Number of rows and columns
print("Shape:", df.shape)
# Display column names
print("Columns:")
print(df.columns)
# Display data types
print("Data Types:")
print(df.dtypes)
# Detailed information about the dataset
df.info()
# Statistical summary of numerical columns
print(df.describe())

                                    # STEP 4: Select Columns
                                    
# Display one column
print(df["name"].head())
# Display multiple columns
print(df[["name", "location", "rate", "votes"]].head())

                                    # STEP 5: Clean Rating Column

# Remove "/5" from ratings such as "4.1/5"
df["rate"] = df["rate"].str.replace("/5", "", regex=False)
# Convert rating from text to numerical value
# Invalid values such as "NEW" become NaN
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
print(df["rate"].head())
print("Rating Data Type:", df["rate"].dtype)

                                # STEP 6: Clean Cost Column

# Store column name in a variable so we don't repeatedly type it
cost_column = "approx_cost(for two people)"
# Remove commas such as "1,200" → "1200"
df[cost_column] = df[cost_column].str.replace(
    ",", "", regex=False
)
# Convert cost to numerical values
df[cost_column] = pd.to_numeric(
    df[cost_column],
    errors="coerce"
)
print(df[cost_column].head())
print("Cost Data Type:", df[cost_column].dtype)

                                        # STEP 7: Filtering

# Display restaurants with rating greater than 4
high_rated = df[df["rate"] > 4]
print("Highly Rated Restaurants:")
print(high_rated.head(10))
# Display restaurants from a specific location
# Change "Banashankari" to a location in your dataset
location_data = df[df["location"] == "Banashankari"]
print("Restaurants in Banashankari:")
print(location_data.head())

                                        # STEP 8: Sorting

# Sort restaurants from highest rating to lowest
top_rated = df.sort_values(
    "rate",
    ascending=False
)
print("Top Rated Restaurants:")
print(top_rated[["name", "rate"]].head(10))
# Sort restaurants according to votes
top_voted = df.sort_values(
    "votes",
    ascending=False
)
print("Most Voted Restaurants:")
print(top_voted[["name", "votes"]].head(10))

                                    # STEP 9: Basic Aggregation

print("Average Rating:", df["rate"].mean())
print("Median Rating:", df["rate"].median())
print("Minimum Rating:", df["rate"].min())
print("Maximum Rating:", df["rate"].max())
print("Average Cost:", df[cost_column].mean())
print("Maximum Cost:", df[cost_column].max())
# Number of unique locations
print("Unique Locations:", df["location"].nunique())
# Number of unique cuisines
print("Unique Cuisines:", df["cuisines"].nunique())

                                    # STEP 10: Feature Engineering

# Function to convert rating into categories
def rating_category(rating):
    if pd.isna(rating):
        return "Unknown"

    elif rating >= 4.5:
        return "Excellent"

    elif rating >= 4:
        return "Good"

    elif rating >= 3:
        return "Average"

    else:
        return "Poor"

# Apply function to every rating
df["Rating_Category"] = df["rate"].apply(
    rating_category
)
print(
    df[
        ["name", "rate", "Rating_Category"]
    ].head(10)
)

                                        
                                        # STEP 11: NumPy

# Create Popularity column using np.where()
# 1000 or more votes → Popular
# Less than 1000 → Less Popular
df["Popularity"] = np.where(
    df["votes"] >= 1000,
    "Popular",
    "Less Popular"
)
print(
    df[
        ["name", "votes", "Popularity"]
    ].head(10)
)


                                    # STEP 12: Univariate EDA

# Analyze rating
print("Rating Mean:", df["rate"].mean())
print("Rating Median:", df["rate"].median())
print("Rating Minimum:", df["rate"].min())
print("Rating Maximum:", df["rate"].max())
# Analyze votes
print("Votes Mean:", df["votes"].mean())
print("Votes Median:", df["votes"].median())
# Analyze cost
print("Cost Mean:", df[cost_column].mean())
print("Cost Median:", df[cost_column].median())

                                    # STEP 13: Value Counts

# Count restaurants in each location
print("Top Locations:")
print(
    df["location"]
    .value_counts()
    .head(10)
)
# Count cuisines
print("Top Cuisines:")
print(
    df["cuisines"]
    .value_counts()
    .head(10)
)

                                        # STEP 14: GROUPBY

# Average rating for every location
average_rating_location = (
    df.groupby("location")["rate"]
    .mean()
    .sort_values(ascending=False)
)
print("Average Rating by Location:")
print(average_rating_location.head(10))
# Average cost for every location
average_cost_location = (
    df.groupby("location")[cost_column]
    .mean()
    .sort_values(ascending=False)
)
print("Average Cost by Location:")
print(average_cost_location.head(10))
# Number of restaurants in every location
restaurant_count = (
    df.groupby("location")["name"]
    .count()
    .sort_values(ascending=False)
)
print("Restaurants by Location:")
print(restaurant_count.head(10))

                                    # STEP 15: Missing Values

# Count missing values in every column
print("Missing Values:")
print(df.isnull().sum())
# Calculate missing-value percentage
missing_percentage = (
    df.isnull().sum() / len(df)
) * 100
print("Missing Percentage:")
print(missing_percentage)

                                    # STEP 16: Mean Imputation

# Calculate average cost
cost_mean = df[cost_column].mean()
# Fill missing Cost values using the mean
df[cost_column] = df[cost_column].fillna(
    cost_mean
)
print(
    "Missing Cost After Mean:",
    df[cost_column].isnull().sum()
)

                                # STEP 17: Median Imputation

# Calculate median cost
cost_median = df[cost_column].median()

print("Median Cost:", cost_median)
# If you want to use median instead of mean:
# df[cost_column] = df[cost_column].fillna(cost_median)

                                # STEP 18: Mode Imputation

# Find the most common cuisine
cuisine_mode = df["cuisines"].mode()[0]
print("Most Common Cuisine:", cuisine_mode)
# Fill missing cuisines with the mode
df["cuisines"] = df["cuisines"].fillna(
    cuisine_mode
)
print(
    "Missing Cuisine After Mode:",
    df["cuisines"].isnull().sum()
)

                                # STEP 19: Forward Fill

# Fill missing rating using the previous available rating
df["rate"] = df["rate"].ffill()
print(
    "Missing Rating After Forward Fill:",
    df["rate"].isnull().sum()
)

                                    # STEP 20: Duplicates

# Count duplicate rows
print(
    "Duplicates Before:",
    df.duplicated().sum()
)
# Display duplicate rows
print(
    df[df.duplicated()]
)
# Remove duplicate rows
df = df.drop_duplicates()
# Check again
print(
    "Duplicates After:",
    df.duplicated().sum()
)

                                    # STEP 21: BIVARIATE EDA

# Scatter plot: Cost vs Rating
plt.scatter(
    df[cost_column],
    df["rate"]
)
plt.xlabel("Cost")
plt.ylabel("Rating")
plt.title("Cost vs Rating")
plt.show()

                                # STEP 22: TOP 10 CUISINES - BAR PLOT

# Find top 10 cuisines
top_cuisines = (
    df["cuisines"]
    .value_counts()
    .head(10)
)
# Create bar chart
plt.bar(
    top_cuisines.index,
    top_cuisines.values
)
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.title("Top 10 Cuisines")
# Rotate names so they are easier to read
plt.xticks(rotation=45)
plt.show()

                                    # STEP 23: PANDAS PROFILING

# Create profiling report
profile = ProfileReport(
    df,
    title="Zomato Dataset Profiling Report"
)

# Save report as HTML
profile.to_file("report.html")

                                # STEP 24: IQR OUTLIER DETECTION

# Q1 = 25th percentile
Q1 = df[cost_column].quantile(0.25)
# Q3 = 75th percentile
Q3 = df[cost_column].quantile(0.75)
# IQR = Q3 - Q1
IQR = Q3 - Q1
# Calculate lower and upper boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

                                        # STEP 25: FIND OUTLIERS

# Values below lower bound OR above upper bound
outliers = df[
    (df[cost_column] < lower_bound) |
    (df[cost_column] > upper_bound)
]

print("Number of Outliers:", len(outliers))
# Display useful information about outliers
print(
    outliers[
        [
            "name",
            "location",
            "cuisines",
            cost_column,
            "rate",
            "votes"
        ]
    ]
)

                                    # STEP 26: FINAL CHECK

print("\n========== FINAL CHECK ==========")
print("Final Shape:", df.shape)
print(
    "Total Missing Values:",
    df.isnull().sum().sum()
)
print(
    "Total Duplicate Rows:",
    df.duplicated().sum()
)
print("\nFinal Data Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
print(df.head())

                                # STEP 27: SAVE CLEANED DATA

# Save cleaned DataFrame as a new CSV
# Original zomato.csv will remain unchanged.
df.to_csv(
    "zomato_cleaned.csv",
    index=False
)
print("\nCleaned dataset saved successfully!")