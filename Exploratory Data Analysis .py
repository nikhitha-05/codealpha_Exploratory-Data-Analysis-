# Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset.csv")  # Replace with your dataset file

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histograms for Numerical Columns
df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions")
plt.show()

# Boxplots for Outlier Detection
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

print("\nEDA Completed Successfully!")
