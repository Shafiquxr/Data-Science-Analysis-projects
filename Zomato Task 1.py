import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------
df = pd.read_csv("zomato.csv.zip")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------------------
# STEP 2: Basic Info
# -----------------------------------------
print(df.info())
print(df.describe())
print(df.isnull().sum())

# -----------------------------------------
# STEP 3: Clean Column Names
# -----------------------------------------
df.drop_duplicates(inplace=True)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------------------
# STEP 4: Clean 'approx_cost(for_two_people)'
# -----------------------------------------
# Convert to string
df['approx_cost(for_two_people)'] = df['approx_cost(for_two_people)'].astype(str)

# Remove commas
df['approx_cost(for_two_people)'] = df['approx_cost(for_two_people)'].str.replace(",", "")

# Convert to numeric (replace invalid values with NaN)
df['approx_cost(for_two_people)'] = pd.to_numeric(
    df['approx_cost(for_two_people)'],
    errors='coerce'
)

# Fill missing cost with median
df['approx_cost(for_two_people)'].fillna(
    df['approx_cost(for_two_people)'].median(),
    inplace=True
)

# -----------------------------------------
# STEP 5: Clean 'rate' / 'aggregate_rating'
# -----------------------------------------
# Zomato rate format like "4.1/5" or "NEW"
df['rate'] = df['rate'].astype(str)
df['rate'] = df['rate'].str.replace("/5", "")
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Create cleaned rating column
df['aggregate_rating'] = df['rate'].fillna(df['rate'].median())
df['aggregate_rating'] = df['aggregate_rating'].clip(0, 5)

# -----------------------------------------
# STEP 6: Clean other columns
# -----------------------------------------
df.fillna({
    "cuisines": "Unknown",
    "location": "Unknown",
    "rest_type": "Unknown"
}, inplace=True)

# -----------------------------------------
# STEP 7: Data Analysis
# -----------------------------------------
print("\nTop 10 Cuisines:")
print(df['cuisines'].value_counts().head(10))

print("\nTop Rated Restaurants:")
top_rated = df[df['aggregate_rating'] > 4.5]
print(top_rated[['name', 'aggregate_rating']].head())

# Cost by city (if column exists)
if "city" in df.columns:
    print("\nAverage Cost by City:")
    print(df.groupby("city")["approx_cost(for_two_people)"]
          .mean()
          .sort_values(ascending=False))

# -----------------------------------------
# STEP 8: Visualizations
# -----------------------------------------

# 1. Distribution of Ratings
plt.figure(figsize=(8,5))
sns.histplot(df['aggregate_rating'], bins=20, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# 2. Top 10 Cuisines
plt.figure(figsize=(10,6))
df['cuisines'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Count")
plt.show()

# 3. Cost vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df['approx_cost(for_two_people)'],
    y=df['aggregate_rating']
)
plt.title("Cost vs Rating")
plt.show()

# -----------------------------------------
# STEP 9: Basic Machine Learning Model
# -----------------------------------------

# Create label: High rating = 1, Low = 0
df["rating_label"] = np.where(df["aggregate_rating"] >= 4.0, 1, 0)

# Features
X = df[["approx_cost(for_two_people)"]]
y = df["rating_label"]

# Train / Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
