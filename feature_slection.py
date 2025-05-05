import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import os

# Set your working directory
os.chdir('/Users/1qxie/Desktop')

# ----------------------------------------
# Step 1: Load raw feature matrix
# ----------------------------------------
df = pd.read_csv("feature_matrix.csv")

# Step 2: Create binary target label from 'fully_funded'
df['label'] = df['fully_funded'].map({'t': 1, 'f': 0})

# Step 3: Filter for Pennsylvania (PA) only
df_pa = df[df['school_state_PA'] == 1]

# Step 4: Restore One-Hot encoded variables into single categorical columns
onehot_groups = {
    'primary_focus_subject_': 'primary_focus_subject',
    'primary_focus_area_': 'primary_focus_area',
    'resource_type_': 'resource_type',
    'grade_level_': 'grade_level',
    'poverty_level_': 'poverty_level',
    'school_metro_': 'school_metro'
}

for prefix, new_col in onehot_groups.items():
    cols = [col for col in df_pa.columns if col.startswith(prefix)]
    df_pa[new_col] = df_pa[cols].idxmax(axis=1).str.replace(prefix, '')
    df_pa.drop(columns=cols, inplace=True)

# Step 5: Define feature set after One-Hot restoration
final_features = [
    'students_reached', 'total_price_excluding_optional_support',
    'avg_unit_price', 'max_unit_price', 'min_unit_price',
    'total_quantity', 'unique_items', 'posted_month', 'posted_dayofweek',
    'school_metro', 'poverty_level', 'grade_level',
    'primary_focus_area', 'primary_focus_subject', 'resource_type',
    'school_charter_t', 'school_magnet_t', 'label'
]

df_final = df_pa[final_features]
df_final.to_csv("pa_with_label_categorical.csv", index=False)

# ----------------------------------------
# Step 6: Load cleaned dataset and separate features/label
# ----------------------------------------
df = pd.read_csv("pa_with_label_categorical.csv")
X = df.drop(columns=["label"])
y = df["label"]

# Step 7: Encode categorical variables
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Step 8: Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Step 9: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X_scaled_df)
retained_columns = X.columns[selector.get_support()]
X_reduced_df = pd.DataFrame(X_reduced, columns=retained_columns)

# Step 10: Train Random Forest and evaluate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_reduced_df, y)
importances = rf.feature_importances_

# Step 11: Get Top 20 most important features
feature_importance = pd.Series(importances, index=retained_columns)
top_20 = feature_importance.sort_values(ascending=False).head(20)

print("Top 20 Most Important Features:")
print(top_20)

# Step 12: Select final Top 17 features (based on visual review)
selected_features = [
    'total_price_excluding_optional_support',
    'max_unit_price',
    'avg_unit_price',
    'min_unit_price',
    'students_reached',
    'posted_month',
    'posted_dayofweek',
    'total_quantity',
    'primary_focus_subject',
    'unique_items',
    'primary_focus_area',
    'school_metro',
    'grade_level',
    'resource_type',
    'poverty_level',
    'school_charter_t',
    'school_magnet_t',
    'label'
]

df_selected = df[selected_features]
df_selected.to_csv("pa_features_top17.csv", index=False)
