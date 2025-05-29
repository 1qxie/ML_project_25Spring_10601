import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load data
def load_data():
    train = pd.read_csv('/Users/1qxie/Desktop/project/data/train_80.csv')
    test = pd.read_csv('/Users/1qxie/Desktop/project/data/test_20.csv')

    for df in [train, test]:
        for col in ['school_charter_t', 'school_magnet_t']:
            df[col] = df[col].astype(str).str.upper().str.strip() == 'TRUE'
    return train, test

train, test = load_data()

# Feature engineering
def enhanced_feature_engineering(df):
    df = df.copy()
    df['students_reached'] = df['students_reached'].replace(0, 1)
    df['price_per_student'] = df['total_price_excluding_optional_support'] / df['students_reached']
    df['price_range'] = df['max_unit_price'] - df['min_unit_price']
    df['price_variability'] = df['avg_unit_price'] / (df['max_unit_price'] + 1e-6)
    df['price_to_student_ratio'] = df['total_price_excluding_optional_support'] / (df['students_reached'] + 1)
    df['item_complexity'] = df['total_quantity'] / (df['unique_items'] + 1)
    df['posted_month_sin'] = np.sin(2 * np.pi * df['posted_month'] / 12)
    df['posted_month_cos'] = np.cos(2 * np.pi * df['posted_month'] / 12)
    df['high_poverty_urban'] = (df['poverty_level'] == 'highest poverty') & (df['school_metro'] == 'urban')
    return df

train = enhanced_feature_engineering(train)
test = enhanced_feature_engineering(test)

# Define features
categorical_cols = [
    'primary_focus_subject', 'primary_focus_area',
    'school_metro', 'grade_level', 'resource_type',
    'poverty_level', 'school_charter_t', 'school_magnet_t'
]

numeric_features = [
    'price_per_student', 'students_reached',
    'posted_month_sin', 'posted_month_cos',
    'price_range', 'price_variability',
    'price_to_student_ratio', 'item_complexity'
]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # ✅ fixed here
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_cols)
])

# Prepare data
X_train = train.drop('label', axis=1)
y_train = train['label']
X_test = test.drop('label', axis=1)
y_test = test['label']

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost Pipeline and GridSearch
pos_count = sum(y_train == 1)
neg_count = sum(y_train == 0)
scale_pos_weight = neg_count / pos_count

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6, 9],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

grid_search_xgb = GridSearchCV(
    xgb_pipeline,
    param_grid=param_grid_xgb,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)
print("Best XGBoost Parameters:", grid_search_xgb.best_params_)
print("Best XGBoost CV ROC AUC:", grid_search_xgb.best_score_)

# RandomForest Pipeline and GridSearch
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [6, 12, None],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search_rf = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid_rf,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train, y_train)
print("Best RandomForest Parameters:", grid_search_rf.best_params_)
print("Best RandomForest CV ROC AUC:", grid_search_rf.best_score_)

# Predict and Evaluate
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nClassification Report - {model_name}")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix - {model_name}")
    print(confusion_matrix(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot PR Curve
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluation
evaluate_model(grid_search_xgb.best_estimator_, X_test, y_test, 'XGBoost')
evaluate_model(grid_search_rf.best_estimator_, X_test, y_test, 'RandomForest')


# Generate prediction outputs and visualization (XGBoost only)
def generate_outputs_and_visuals(model, X_test, y_test, original_test_df):
    y_proba = model.predict_proba(X_test)[:, 1]
    original_test_df = original_test_df.copy()
    original_test_df['pred_prob'] = y_proba

    # Save top prediction results
    output_cols = [
        'project_id', 'price_per_student', 'students_reached',
        'primary_focus_subject', 'poverty_level', 'pred_prob'
    ]
    original_test_df[output_cols].sort_values(by='pred_prob', ascending=False).to_csv(
        'xgb_test_prediction.csv', index=False
    )

    # Plot histogram of prediction scores
    plt.figure(figsize=(8, 5))
    sns.histplot(y_proba, bins=25, color='skyblue', edgecolor='black')
    plt.axvline(0.8, color='red', linestyle='--', label='High Confidence Threshold (0.8)')
    plt.title('Prediction Score Distribution (Funded Probability)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of Projects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Show top N high-confidence predictions
    top_projects = original_test_df.sort_values(by='pred_prob', ascending=False).head(10)
    print("\nTop 10 High-Confidence Projects (Predicted Funded):")
    print(top_projects[output_cols])

    # Calibration Curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run on XGBoost
original_test_df = test.copy()
original_test_df = original_test_df.reset_index(drop=True)
original_test_df['project_id'] = ['P{}'.format(i) for i in range(len(original_test_df))]
generate_outputs_and_visuals(grid_search_xgb.best_estimator_, X_test, y_test, original_test_df)

def precision_at_k(y_true, y_scores, k):
    """
    Compute Precision@K: Among top K predicted scores, how many are truly positive.
    """
    k = min(k, len(y_scores))
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    top_k_true = np.array(y_true)[sorted_indices]
    precision = np.sum(top_k_true) / k
    return precision

# Run Precision@K
y_proba = grid_search_xgb.best_estimator_.predict_proba(X_test)[:, 1]
precision_50 = precision_at_k(y_test, y_proba, 50)
precision_100 = precision_at_k(y_test, y_proba, 100)

print(f"Precision@50: {precision_50:.2f}")
print(f"Precision@100: {precision_100:.2f}")
