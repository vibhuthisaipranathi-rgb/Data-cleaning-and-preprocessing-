# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Basic info
print("Initial Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# 2. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)  # too many missing values

# 3. Encoding categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# 4. Normalize/Standardize numerical features
scaler = StandardScaler()
numeric_features = ['Age', 'Fare']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 5. Outlier detection and removal using boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(data=df[numeric_features])
plt.title("Boxplot of Numerical Features")
plt.show()

# Remove outliers (example: using IQR for 'Fare')
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# Final cleaned data info
print("\nCleaned Data Info:")
print(df.info())
print(df.head())
