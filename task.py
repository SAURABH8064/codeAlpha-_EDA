import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Load
data = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\codealpha_ EDA folder\titanic.csv")  # अगर आपकी फाइल का नाम Titanic-Dataset.csv है तो वही लिखें

# First 5 Rows
print("First 5 Rows")
print(data.head())

# Dataset Information
print("\nDataset Information")
print(data.info())

# Missing Values
print("\nMissing Values")
print(data.isnull().sum())

# Summary Statistics
print("\nSummary Statistics")
print(data.describe())

# Survival Count
print("\nSurvival Count")
print(data["Survived"].value_counts())

# Gender Count
print("\nGender Count")
print(data["Sex"].value_counts())

# Passenger Class Count
print("\nPassenger Class")
print(data["Pclass"].value_counts())

# Age Distribution
plt.figure(figsize=(6,4))
plt.hist(data["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=data)
plt.title("Survival by Gender")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
numeric = data.select_dtypes(include="number")
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")