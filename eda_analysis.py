import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
df = pd.read_csv("titanic.csv", encoding="latin1")

print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:\n", df.describe(include='all'))

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# ----------------------------
# 1. Distribution Analysis
# ----------------------------

fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.histplot(df['Age'], bins=30, kde=True, ax=ax[0])
ax[0].set_title("Age Distribution")

sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax[1])
ax[1].set_title("Fare by Passenger Class")

plt.show()

# Categorical analysis
plt.figure(figsize=(8,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# ----------------------------
# 2. Correlation Analysis
# ----------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Cross-tabulation
print("\nSurvival Rate by Class:")
print(pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100)

# ----------------------------
# 3. Outlier Detection
# ----------------------------

plt.figure(figsize=(8,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# Z-score method
z_scores = np.abs(stats.zscore(df['Fare']))
outliers = df[z_scores > 3]

print("Number of Fare Outliers:", len(outliers))

# ----------------------------
# 4. Advanced Visualizations
# ----------------------------

g = sns.FacetGrid(df, col='Survived', row='Pclass', height=3)
g.map(sns.histplot, 'Age', bins=20)
plt.show()

sns.pairplot(df[['Age','Fare','Parch','Survived']], hue='Survived')
plt.show()

print("\nEDA Completed Successfully")