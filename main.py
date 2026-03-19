import pandas as pd

# Load Cleveland Heart Disease dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=column_names, na_values='?')

# Convert target: any value > 0 becomes 1 (heart disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"Dataset shape: {df.shape}")
print(f"Diseased: {df['target'].sum()}, Healthy: {(df['target']==0).sum()}")
print(df.head())

# Check missing values
print("\n=== Missing Values Before Filtering ===")
print(df.isnull().sum())

# Replace missing values with column mean (Filtering Technique)
# Formula: μ = Σxi / n
for col in ['ca', 'thal']:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)
    print(f"{col}: null values replaced with mean = {mean_val:.4f}")

# Verify no nulls remain
print(f"\nTotal missing values after filtering: {df.isnull().sum().sum()}")

# Discretization - convert numeric features to categorical values
def discretize_age(val):
    if val <= 12:   return 0  # Child
    elif val <= 18: return 1  # Teenager
    elif val <= 59: return 2  # Adult
    else:           return 3  # Senior Adult

def discretize_trestbps(val):
    if val < 120:   return 0  # Ideal
    elif val < 130: return 1  # Normal
    elif val < 140: return 2  # Up-Normal
    else:           return 3  # High

def discretize_chol(val):
    if val < 200:   return 0  # Optimal
    elif val < 240: return 1  # Desirable
    elif val < 280: return 2  # High
    else:           return 3  # Very High

def discretize_thalach(val):
    if val < 111:   return 0  # Low
    elif val <= 141: return 1  # Medium
    else:            return 2  # High

def discretize_oldpeak(val):
    if val < 2:   return 0  # Low
    elif val < 4: return 1  # Medium
    else:         return 2  # High

# Apply discretization
df['age']      = df['age'].apply(discretize_age)
df['trestbps'] = df['trestbps'].apply(discretize_trestbps)
df['chol']     = df['chol'].apply(discretize_chol)
df['thalach']  = df['thalach'].apply(discretize_thalach)
df['oldpeak']  = df['oldpeak'].apply(discretize_oldpeak)

# Round ca and thal (were replaced by float mean)
df['ca']   = df['ca'].round().astype(int)
df['thal'] = df['thal'].round().astype(int)

print("=== Dataset After Discretization ===")
print(df.head(10))
print(f"\nData types:\n{df.dtypes}")

# Feature Selection using Naïve Bayes
def compute_prior(df, target='target'):
    total = len(df)
    return df[target].value_counts().to_dict()  
    return {c: count/total for c, count in counts.items()}

def feature_selection_nb(df, target='target'):
    features = [col for col in df.columns if col != target]
    total = len(df)
    
    # P(c) - prior probability
    p_c = {c: count/total for c, count in df[target].value_counts().items()}
    
    feature_scores = {f: 0.0 for f in features}
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f1, f2 = features[i], features[j]
            dependency = 0.0
            
            for c in df[target].unique():
                subset_c = df[df[target] == c]
                
                # P(f1|c) and P(f2|c)
                for v1 in df[f1].unique():
                    p_f1_c = len(subset_c[subset_c[f1] == v1]) / len(subset_c)
                    p_f1   = len(df[df[f1] == v1]) / total
                    
                    for v2 in df[f2].unique():
                        p_f2_c = len(subset_c[subset_c[f2] == v2]) / len(subset_c)
                        p_f2   = len(df[df[f2] == v2]) / total
                        
                        denom = p_f1 + p_f2
                        if denom > 0:
                            dependency += (p_c[c] * p_f1_c * p_f2_c) / denom
            
            feature_scores[f1] += dependency
            feature_scores[f2] += dependency
    
    # Sort by score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("=== Feature Ranking by Naïve Bayes Dependency ===")
    for rank, (feat, score) in enumerate(sorted_features, 1):
        marker = " <-- SELECTED" if rank <= 6 else ""
        print(f"{rank:2d}. {feat:12s}: {score:.4f}{marker}")
    
    selected = [feat for feat, _ in sorted_features[:6]]
    print(f"\nSelected features (B): {selected}")
    return selected

selected_features = feature_selection_nb(df)

# Note: The paper explicitly states the selected features as:
# B = {age, sex, trestbps, fbs, chol, exang}
# We use the paper's defined subset for the classification phase
selected_features = ['age', 'sex', 'trestbps', 'fbs', 'chol', 'exang']
print(f"Features used for classification (as per paper): {selected_features}")

