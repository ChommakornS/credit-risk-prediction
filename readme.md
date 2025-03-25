# Credit Default Prediction

## Project Overview
This project aims to predict the likelihood of credit card default using machine learning models. The dataset used is the [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) from the UCI Machine Learning Repository, which contains information on default payments, demographic factors, credit data, payment history, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Class Imbalance Handling](#class-imbalance-handling)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Feature Importance Analysis](#feature-importance-analysis)
8. [Statistical Analysis](#statistical-analysis)
9. [Hypothesis Testing](#hypothesis-testing)
10. [A/B Testing Simulation](#ab-testing-simulation)
11. [Time Series Analysis](#time-series-analysis)
12. [Hyperparameter Tuning](#hyperparameter-tuning)
13. [Model Comparison and Threshold Optimization](#model-comparison-and-threshold-optimization)
14. [Results and Conclusions](#results-and-conclusions)

## Dataset Description
The dataset contains 25 variables:
- **ID**: ID of each client
- **LIMIT_BAL**: Amount of credit given
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level (1=graduate school, 2=university, 3=high school, 4=others)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: History of past payment (PAY_0 = most recent)
  - Values: -2 = no consumption; -1 = paid in full; 0 = revolving credit; 1-9 = months of delay
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts
- **PAY_AMT1 to PAY_AMT6**: Previous payment amounts
- **default**: Default payment (1=yes, 0=no)

## Data Preprocessing

### Data Loading and Cleaning
- **Error Handling**: Implemented robust file loading with error catching and fallback for different encodings:
  ```python
  try:
      df = pd.read_csv(file_path)
  except Exception as e:
      print(f"Error loading CSV file: {e}")
      print("Trying with different encoding...")
      try:
          df = pd.read_csv(file_path, encoding='latin1')
      except Exception as e2:
          print(f"Still failed with error: {e2}")
          return None
  ```
  
- **Column Name Standardization**: Detects and handles different column naming conventions:
  ```python
  # Case 1: Check if the file has X1, X2, etc. columns
  if 'X1' in df.columns:
      print("Detected X1, X2, etc. column format. Renaming columns...")
      column_names = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                      'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                      'default']
      # [...]
  ```

- **Missing Value Detection and Reporting**: Systematically identifies and reports missing values:
  ```python
  missing_values = df.isnull().sum()
  print("\nMissing values in each column:")
  print(missing_values)
  ```

### Data Type Conversion and Cleaning
- **Type Conversion**: Ensures all columns have appropriate data types for analysis:
  ```python
  for col in processed_df.columns:
      if processed_df[col].dtype == 'object':
          print(f"Converting column '{col}' to numeric")
          processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
  ```

- **Non-Numeric Column Handling**: Identifies and handles non-numeric columns:
  ```python
  non_numeric_cols = [col for col in processed_df.columns if processed_df[col].dtype == 'object']
  if non_numeric_cols:
      print(f"Warning: The following columns could not be converted to numeric: {non_numeric_cols}")
      print("These columns will be dropped.")
      processed_df = processed_df.drop(columns=non_numeric_cols)
  ```

### Missing Value Imputation
- **Median Imputation for Numeric Features**: Replaces missing values with median values:
  ```python
  for col in processed_df.columns:
      if processed_df[col].isna().sum() > 0:
          # Fill numeric columns with median
          if pd.api.types.is_numeric_dtype(processed_df[col]):
              median_val = processed_df[col].median()
              processed_df[col] = processed_df[col].fillna(median_val)
              print(f"  - Filled with median: {median_val}")
  ```

- **Mode Imputation for Categorical Features**: Used for categorical columns with missing values:
  ```python
  # Fill NaNs with mode
  if X[col].isna().any():
      mode_val = X[col].mode()[0]
      X[col] = X[col].fillna(mode_val)
  ```

### Categorical Variable Encoding
- **Binary Encoding**: Converts multi-level categorical variables to binary:
  ```python
  # Sex: 1=male, 2=female -> convert to binary 0=male, 1=female
  if 'SEX' in processed_df.columns:
      processed_df['SEX'] = processed_df['SEX'].map({1: 0, 2: 1})
  
  # Marriage: 1=married, 2=single, 3=others -> simplify to binary: 0=not married, 1=married
  if 'MARRIAGE' in processed_df.columns:
      processed_df['MARRIAGE'] = processed_df['MARRIAGE'].map({1: 1, 2: 0, 3: 0, 0: 0})
  ```

- **Ordinal Encoding**: Preserves order for education level:
  ```python
  # Education: 1=graduate school, 2=university, 3=high school, 4=others
  # Simplify to: 0=high school or below, 1=university, 2=graduate school
  if 'EDUCATION' in processed_df.columns:
      education_mapping = {1: 2, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 0: 0}
      processed_df['EDUCATION'] = processed_df['EDUCATION'].map(education_mapping)
  ```

## Feature Engineering

### Financial Metrics
- **Aggregation Features**: Creates total bill and payment amounts:
  ```python
  # 1. Total bill amount and payment amount
  processed_df['TOTAL_BILL_AMT'] = processed_df[bill_cols].sum(axis=1)
  processed_df['TOTAL_PAY_AMT'] = processed_df[pay_cols].sum(axis=1)
  ```

- **Ratio Features**: Calculates utilization and payment ratios with careful handling of edge cases:
  ```python
  # 2. Utilization ratio (total bill / credit limit)
  mask = processed_df['LIMIT_BAL'] > 0
  processed_df['UTILIZATION_RATIO'] = 0.0  # Default value
  processed_df.loc[mask, 'UTILIZATION_RATIO'] = processed_df.loc[mask, 'TOTAL_BILL_AMT'] / processed_df.loc[mask, 'LIMIT_BAL']
  
  # 3. Payment ratio (total payment / total bill) where bill > 0
  mask = processed_df['TOTAL_BILL_AMT'] > 0
  processed_df['PAYMENT_RATIO'] = 0.0  # Default value
  processed_df.loc[mask, 'PAYMENT_RATIO'] = processed_df.loc[mask, 'TOTAL_PAY_AMT'] / processed_df.loc[mask, 'TOTAL_BILL_AMT']
  ```

- **Statistical Features**: Creates features capturing min, max, and variability:
  ```python
  # 4. Max bill amount
  processed_df['MAX_BILL_AMT'] = processed_df[bill_cols].max(axis=1)
  
  # 5. Min bill amount
  processed_df['MIN_BILL_AMT'] = processed_df[bill_cols].min(axis=1)
  
  # 6. Bill amount standard deviation
  processed_df['BILL_AMT_STD'] = processed_df[bill_cols].std(axis=1)
  ```

### Payment Behavior Metrics
- **Count-Based Features**: Counts instances of delays:
  ```python
  # 7. Count of delays
  processed_df['DELAY_COUNT'] = (processed_df[delay_cols] > 0).sum(axis=1)
  
  # 8. Max delay
  processed_df['MAX_DELAY'] = processed_df[delay_cols].max(axis=1)
  ```

- **Weighted Features**: Creates weighted score giving higher importance to recent delays:
  ```python
  # 9. Weighted delay score
  # Sort delay columns to ensure correct ordering
  delay_cols_sorted = sorted(delay_cols, 
                           key=lambda x: int(x.replace('PAY_', '')) if x != 'PAY_0' else 0)
  
  # Apply decreasing weights (most recent month gets highest weight)
  weights = list(range(1, len(delay_cols) + 1))
  weights.reverse()  # Reverse to give higher weight to recent months
  
  processed_df['WEIGHTED_DELAY_SCORE'] = 0
  for feature, weight in zip(delay_cols_sorted, weights):
      # Only positive values indicate delay (clip negative values to 0)
      delay_value = processed_df[feature].clip(lower=0)
      processed_df['WEIGHTED_DELAY_SCORE'] += delay_value * weight
  ```

### Demographic Features
- **Binning**: Creates age groups using pandas cut function:
  ```python
  # 10. Age groups
  processed_df['AGE_GROUP'] = pd.cut(processed_df['AGE'], 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=[0, 1, 2, 3, 4])
  processed_df['AGE_GROUP'] = processed_df['AGE_GROUP'].astype('float').astype('Int64')
  ```

## Exploratory Data Analysis

### Target Distribution Analysis
- **Value Counting**: Analyzes the distribution of the target variable:
  ```python
  default_counts = df[target_col].value_counts()
  default_perc = df[target_col].value_counts(normalize=True) * 100
  ```

- **Visualization**: Creates visual representation of class distribution:
  ```python
  plt.pie(default_counts, labels=labels, autopct='%1.1f%%', 
          colors=colors[:len(default_counts)], explode=explode)
  plt.title('Distribution of Target Variable', fontsize=16)
  ```

### Feature Distribution Analysis
- **Numerical Feature Visualization**: Creates histograms with kernel density estimates:
  ```python
  plt.figure(figsize=(15, 10))
  for i, feature in enumerate(numerical_features):
      plt.subplot(2, 2, i+1)
      sns.histplot(data=df, x=feature, hue='default', kde=True, bins=30)
      plt.title(f'Distribution of {feature} by Default Status')
  ```

- **Categorical Feature Visualization**: Creates count plots for categorical variables:
  ```python
  plt.figure(figsize=(15, 10))
  for i, feature in enumerate(categorical_features):
      plt.subplot(2, 2, i+1)
      sns.countplot(data=df, x=feature, hue='default')
      plt.title(f'Distribution of {feature} by Default Status')
  ```

### Correlation Analysis
- **Correlation Matrix**: Calculates and visualizes feature correlations:
  ```python
  corr_matrix = df.corr()
  plt.figure(figsize=(14, 12))
  mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
  sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=.3, center=0,
              square=True, linewidths=.5, annot=False)
  ```

## Class Imbalance Handling

The project addresses class imbalance through the `handle_imbalance` function, which provides multiple resampling methods:

### Implemented Methods
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic examples of the minority class by generating new instances between existing minority samples
- **Random Undersampling**: Reduces majority class samples to balance the dataset by randomly removing majority class instances
- **Random Oversampling**: Duplicates minority class samples to balance the dataset by randomly duplicating minority class instances
- **None**: Option to retain the original imbalanced dataset if desired

### Implementation Details
```python
def handle_imbalance(X, y, method='smote'):
    """
    Handle class imbalance using different methods.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    method : str
        Method to handle imbalance: 'smote', 'undersampling', 'oversampling', or 'none'
    
    Returns:
    --------
    X_resampled, y_resampled : pandas.DataFrame, pandas.Series
        Resampled feature matrix and target vector
    """
```

The function first prepares the data by:
1. Converting float columns to float32 to avoid precision issues
2. Handling potential NaN values in integer columns by filling them with the mode
3. Converting columns to appropriate types to ensure compatibility with resampling methods

Then it applies the selected resampling technique:
```python
if method.lower() == 'smote':
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

elif method.lower() == 'undersampling':
    from imblearn.under_sampling import RandomUnderSampler
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

elif method.lower() == 'oversampling':
    from imblearn.over_sampling import RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
```

### Application in the Pipeline
The notebook primarily uses SMOTE for handling imbalance:
```python
# Handle class imbalance with SMOTE
X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train, method='smote')
```

This balanced dataset is then used for model training:
```python
logreg_model = train_logistic_regression(X_train_balanced, y_train_balanced, X_test, y_test)
rf_model, feature_importance = train_random_forest(X_train_balanced, y_train_balanced, X_test, y_test)
xgb_model = train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)
```

## Model Training and Evaluation

### Train-Test Split with Stratification
- **Stratified Sampling**: Ensures balanced class distribution in train and test sets:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
  ```

### Pipeline Implementation
- **Scikit-Learn Pipelines**: Creates standardized workflows combining preprocessing and model training:
  ```python
  pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', LogisticRegression(max_iter=1000, random_state=42))
  ])
  ```

### Cross-Validation
- **K-Fold Cross-Validation**: Implements k-fold CV with shuffling:
  ```python
  cv = KFold(n_splits=5, shuffle=True, random_state=42)
  cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
  ```

- **CV Reporting**: Reports distribution of scores across folds:
  ```python
  print(f"Cross-validation ROC-AUC scores: {cv_scores}")
  print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
  ```

### Models Implemented
1. **Logistic Regression**
   ```python
   def train_logistic_regression(X_train, y_train, X_test, y_test):
       # Create a pipeline with preprocessing and model
       pipeline = Pipeline([
           ('scaler', StandardScaler()),
           ('clf', LogisticRegression(max_iter=1000, random_state=42))
       ])
       # [...]
   ```

2. **Random Forest**
   ```python
   def train_random_forest(X_train, y_train, X_test, y_test):
       # Create a pipeline with preprocessing and model
       pipeline = Pipeline([
           ('scaler', StandardScaler()),
           ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
       ])
       # [...]
   ```

3. **XGBoost**
   ```python
   def train_xgboost(X_train, y_train, X_test, y_test):
       # Create a pipeline with preprocessing and model
       pipeline = Pipeline([
           ('scaler', StandardScaler()),
           ('clf', xgb.XGBClassifier(
               n_estimators=100,
               learning_rate=0.1,
               max_depth=5,
               random_state=42,
               use_label_encoder=False,
               eval_metric='logloss'
           ))
       ])
       # [...]
   ```


### Comprehensive Evaluation Metrics
- **Multiple Classification Metrics**: Calculates various performance metrics:
  ```python
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred_proba)
  ```

- **Confusion Matrix Visualization**: Visualizes true vs. predicted classes:
  ```python
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
  ```

- **ROC Curve Analysis**: Plots ROC curve with AUC score:
  ```python
  fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
  plt.plot([0, 1], [0, 1], 'k--')
  ```

- **Classification Report**: Provides precision, recall, f1-score by class:
  ```python
  print("Classification Report:")
  print(classification_report(y_true, y_pred))
  ```

## Feature Importance Analysis

- **Random Forest Feature Importance**: Extracts and visualizes feature importance from random forest model:
  ```python
  rf_model = pipeline.named_steps['clf']
  feature_importance = pd.DataFrame({
      'Feature': X_train.columns,
      'Importance': rf_model.feature_importances_
  }).sort_values('Importance', ascending=False)
  
  print("\nTop 10 most important features:")
  print(feature_importance.head(10))
  
  plt.figure(figsize=(12, 8))
  sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
  ```

## Statistical Analysis

### Descriptive Statistics by Target
- **Group-Based Statistics**: Calculates statistics by default status:
  ```python
  stats_by_default = df.groupby('default')[feature].agg(['mean', 'median', 'std', 'min', 'max'])
  ```

### Statistical Testing
- **T-Tests**: Compares means between default and non-default groups:
  ```python
  default_group = df[df['default'] == 1][feature].dropna()
  non_default_group = df[df['default'] == 0][feature].dropna()
  
  t_stat, p_value = stats.ttest_ind(default_group, non_default_group, equal_var=False)
  ```

- **Chi-Square Tests**: Tests association between categorical variables and default:
  ```python
  contingency_table = pd.crosstab(df[feature], df['default'])
  chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
  ```

### Correlation Analysis with Target
- **Target Correlation Analysis**: Identifies features most correlated with default:
  ```python
  correlation_with_default = numeric_df.corr()['default'].sort_values(ascending=False)
  ```

- **Visualization**: Creates bar plot of top correlations:
  ```python
  top_corr_features = correlation_with_default.drop('default').abs().sort_values(ascending=False).head(15).index
  sns.barplot(x=correlation_with_default[top_corr_features].values, 
              y=top_corr_features, palette='viridis')
  ```

## Hypothesis Testing

The project tests several key hypotheses using formal statistical methods:

### Credit Limit Hypothesis
Tests whether customers with higher credit limits are less likely to default:
```python
# Define high and low credit limit groups
median_limit = df['LIMIT_BAL'].median()
high_limit_default_rate = df[df['LIMIT_BAL'] > median_limit]['default'].mean()
low_limit_default_rate = df[df['LIMIT_BAL'] <= median_limit]['default'].mean()

# Z-test for proportions
counts = np.array([high_limit_defaults, low_limit_defaults])
nobs = np.array([high_limit_count, low_limit_count])

z_stat, p_value = proportions_ztest(counts, nobs)
```

### Payment Delay Hypothesis
Tests whether customers with payment delays have higher default rates:
```python
# Define groups
has_delay = df['DELAY_COUNT'] > 0
delay_default_rate = df[has_delay]['default'].mean()
no_delay_default_rate = df[~has_delay]['default'].mean()

# Statistical testing
counts = np.array([delay_defaults, no_delay_defaults])
nobs = np.array([delay_count, no_delay_count])

z_stat, p_value = proportions_ztest(counts, nobs)
```

### Age Effect Hypothesis
Tests whether age significantly affects default probability:
```python
# Analyze default rates by age group
age_groups = df.groupby('AGE_GROUP')['default'].mean()

# ANOVA test
f_stat, p_value = stats.f_oneway(*[group for group in age_groups_list if len(group) > 0])
```

## A/B Testing Simulation

Implements a simulation comparing two different credit limit strategies:

### Simulation Setup
```python
# Create a random subset for simulation
np.random.seed(42)
sample_indices = np.random.choice(df.index, size=10000, replace=False)
sample_df = df.loc[sample_indices].copy()

# Strategy A: Higher education gets higher credit limits
sample_df['strategy_A'] = np.where(sample_df['EDUCATION'] <= 1, 'high_limit', 'regular_limit')

# Strategy B: Good payment history gets higher credit limits
sample_df['strategy_B'] = np.where(sample_df['DELAY_COUNT'] == 0, 'high_limit', 'regular_limit')
```

### Statistical Analysis
```python
# Z-test for Strategy A
counts_A = np.array([defaults_high_A, defaults_regular_A])
nobs_A = np.array([count_high_A, count_regular_A])
z_stat_A, p_value_A = proportions_ztest(counts_A, nobs_A)

# Z-test for Strategy B
counts_B = np.array([defaults_high_B, defaults_regular_B])
nobs_B = np.array([count_high_B, count_regular_B])
z_stat_B, p_value_B = proportions_ztest(counts_B, nobs_B)
```

### Strategy Comparison
```python
# Compare strategies
if p_value_A < 0.05 and p_value_B < 0.05:
    diff_A = abs(high_limit_A['default'].mean() - regular_limit_A['default'].mean())
    diff_B = abs(high_limit_B['default'].mean() - regular_limit_B['default'].mean())
    
    if diff_B > diff_A:
        print("Strategy B (Payment history-based) shows a larger effect on default rates.")
    else:
        print("Strategy A (Education-based) shows a larger effect on default rates.")
```

## Time Series Analysis

Implements time series techniques for analyzing payment patterns and forecasting:

### Trend Analysis
```python
# Get bill amounts for the last 6 months
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

# Aggregate data for all customers
avg_bills = sample_df[bill_cols].mean()
avg_pays = sample_df[pay_cols].mean()

# Plot time series data
plt.figure(figsize=(12, 6))
plt.plot(months, avg_bills.values, 'o-', label='Average Bill Amount', linewidth=2)
plt.plot(months, avg_pays.values, 's-', label='Average Payment', linewidth=2)
```

### ARIMA Forecasting
```python
# Prepare the time series data
ts_data = avg_bills.values

# Fit ARIMA model
model = ARIMA(ts_data, order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 3 months
forecast = model_fit.forecast(steps=3)
```

### Payment Pattern Analysis
```python
# Create payment pattern features
sample_df['increasing_bills'] = (sample_df['BILL_AMT1'] > sample_df['BILL_AMT2']) & \
                              (sample_df['BILL_AMT2'] > sample_df['BILL_AMT3'])

sample_df['decreasing_payments'] = (sample_df['PAY_AMT1'] < sample_df['PAY_AMT2']) & \
                                  (sample_df['PAY_AMT2'] < sample_df['PAY_AMT3'])

# Calculate default rates by pattern
pattern_default_rates = sample_df.groupby(['increasing_bills', 'decreasing_payments'])['default'].mean()
```

## Hyperparameter Tuning

Implements sophisticated tuning of model hyperparameters:

### RandomizedSearchCV
```python
def hyperparameter_tuning(X_train, y_train, model_type='xgboost'):
    # Define the parameter grid
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__subsample': [0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create the model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
```

### Parameters Tuned
- **XGBoost**:
  - n_estimators: Number of boosting rounds
  - max_depth: Maximum depth of trees
  - learning_rate: Step size shrinkage
  - subsample: Subsample ratio of training instances
  - colsample_bytree: Subsample ratio of columns for each tree


## Model Comparison and Threshold Optimization

### Model Comparison Framework
```python
def compare_models(models_results, X_test, y_test):
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(models_results.keys()),
        'Accuracy': [models_results[model]['metrics']['accuracy'] for model in models_results],
        'Precision': [models_results[model]['metrics']['precision'] for model in models_results],
        'Recall': [models_results[model]['metrics']['recall'] for model in models_results],
        'F1 Score': [models_results[model]['metrics']['f1'] for model in models_results],
        'ROC AUC': [models_results[model]['metrics']['roc_auc'] for model in models_results]
    })
    
    # Sort by ROC AUC
    comparison_df = comparison_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
```
