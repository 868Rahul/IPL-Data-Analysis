import numpy as np
import pandas as pd

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("batting_card.csv")

print("Initial Shape:", df.shape)
print(df.info())

# -----------------------------
# 2. Drop completely useless columns
# -----------------------------
# 'link' column has 0 non-null values → remove it
df.drop(columns=["link"], inplace=True)

# -----------------------------
# 3. Handling Missing Values
# -----------------------------

# season → convert to int and fill missing with mode (most common season)
df['season'] = df['season'].fillna(df['season'].mode()[0]).astype(int)

# These columns have few missing values → fill with 0
cols_fill_zero = ['runs', 'ballsFaced', 'fours', 'sixes']
for col in cols_fill_zero:
    df[col] = df[col].fillna(0)

# runningOver → has many missing → fill with -1 (indicating unavailable)
df['runningOver'] = df['runningOver'].fillna(-1)

# commentary → fill missing with "No Commentary"
df['commentary'] = df['commentary'].fillna("No Commentary")

# -----------------------------
# 4. Fix Wrong Data Types
# -----------------------------

# Convert 'strikeRate' to numeric
df['strikeRate'] = pd.to_numeric(df['strikeRate'], errors='coerce').fillna(0)

# Convert 'isNotOut' from object to boolean
df['isNotOut'] = df['isNotOut'].map({'True': True, 'False': False, 'Yes': True, 'No': False})
df['isNotOut'] = df['isNotOut'].fillna(False)

# Convert 'minutes' to numeric
df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0).astype(int)

# Convert runningScore → numeric
df['runningScore'] = pd.to_numeric(df['runningScore'], errors='coerce').fillna(0).astype(int)

# -----------------------------
# 5. Standardizing Column Names
# -----------------------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# 6. Final Check
# -----------------------------
print("\nAfter Cleaning:")
print(df.info())
print(df.head())

# Save cleaned file
df.to_csv("cleaned_batting_card.csv", index=False)
print("\nCleaned file saved as cleaned_batting_card.csv")
