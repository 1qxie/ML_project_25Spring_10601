import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Set your working directory
os.chdir('/Users/1qxie/Desktop')

# 1. Load the dataset
df = pd.read_csv("pa_features_top17.csv")

# 2. Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# 3. Perform stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Optionally recombine and save
train_df = X_train.copy()
train_df['label'] = y_train
test_df = X_test.copy()
test_df['label'] = y_test

train_df.to_csv("train_80.csv", index=False)
test_df.to_csv("test_20.csv", index=False)

# 5. Print class balance
print("Split completed with label balance:")
print("Train label distribution:\n", y_train.value_counts(normalize=True))
print("Test label distribution:\n", y_test.value_counts(normalize=True))
