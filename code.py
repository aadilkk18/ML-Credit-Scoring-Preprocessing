import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline

# 1. LOAD DATASET
df = pd.read_csv('clean_dataset.csv')

# 2. SELECT RELEVANT CONTINUOUS COLUMNS
# We exclude categorical numbers like Gender, Married, BankCustomer, etc.
relevant_cols = ['Income', 'CreditScore', 'Debt', 'Age', 'YearsEmp']

# Verify columns exist in your specific CSV
available_cols = [col for col in relevant_cols if col in df.columns]
target_col = 'Approved'  # or df.columns[-1]

X = df[available_cols]
y = df[target_col]

# 3. ADVANCED PREPROCESSING PIPELINE
pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('scaler', RobustScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson'))
])

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 5. EXECUTE PIPELINE
X_train_transformed = pipeline.fit_transform(X_train)
X_train_df = pd.DataFrame(X_train_transformed, columns=available_cols)

# 6. VISUALIZE RELEVANT COLUMNS ONE BY ONE
for col in available_cols:
    plt.figure(figsize=(12, 5))

    # Plot Before (Raw Data)
    plt.subplot(1, 2, 1)
    sns.histplot(X_train[col], kde=True, color='salmon')
    plt.title(f'ORIGINAL: {col} (Skewed Distribution)')

    # Plot After (Transformed Data)
    plt.subplot(1, 2, 2)
    sns.histplot(X_train_df[col], kde=True, color='skyblue')
    plt.title(f'TRANSFORMED: {col} (Gaussian/Normal Distribution)')

    plt.tight_layout()
    plt.show()

print(f"--- Processed relevant features: {available_cols} ---")
