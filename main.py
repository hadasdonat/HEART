import pandas as pd
import classification

# ======================================================================================
# Step 1: Selection and structure of the database
# ======================================================================================
def step_1_load_data():
    # 1.1: Load Cleveland Heart Disease dataset from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # 1.2: Define the 14 attributes (13 medical features + 1 target variable)
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # Load dataset and treat '?' as NaN values
    df = pd.read_csv(url, names=column_names, na_values='?')

    # 1.3: Target Binarization
    # According to the paper, any value > 0 indicates heart disease (1), and 0 indicates healthy (0)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    print("--- Step 1: Database Loading and Structure ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Diseased: {df['target'].sum()}, Healthy: {(df['target']==0).sum()}")
    print(df.head())
    return df



# ======================================================================================
# Step 2: Data quality and missing values
# ======================================================================================
def step_2_handle_missing_values(df):
    # 2.3 & 2.4: Scan for null values specifically in 'ca' and 'thal' attributes
    missing_info = df.isnull().sum()
    print("\n--- Step 2: Missing Values Detection and Handling ---")
    print("Missing values found before processing:")
    print(missing_info)

    # 2.5: Replace missing values using the Filtering Technique (Mean Imputation)
    # Formula: μ = Σxi / n
    for col in ['ca', 'thal']:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Fixed {col}: nulls replaced with mean value = {mean_val:.4f}")

    # Final verification to ensure no null values remain
    print(f"\nTotal missing values after filtering: {df.isnull().sum().sum()}")
    return df   


# ======================================================================================
# Step 3: Discretization Process
# ======================================================================================
# Helper Functions for Discretization based on Table 1
def discretize_age(val):
    if val <= 12:   return 0  # Child
    elif val <= 18: return 1  # Adolescence
    elif val <= 59: return 2  # Adult
    else:           return 3  # Senior Adult

def discretize_trestbps(val):
    if val < 120:   return 0  # Ideal
    elif val < 130: return 1  # Normal
    elif val < 140: return 2  # Up-Normal
    else:           return 3  # High

def discretize_chol(val):
    if val < 150:   return 0 #Optimal
    elif val < 200: return 1 #Desirable
    elif val < 400: return 2 #High
    else:           return 3 #Very High

def discretize_thalach(val):
    if val < 111:   return 0  # Low
    elif val <= 141: return 1  # Medium
    else:            return 2  # High

# Note: although figure 4 in the paper shows raw float values for 'oldpeak', the right thing to do is to apply discretization.
def discretize_oldpeak(val):
    """
    Proposed discretization for Oldpeak (0-6.2 domain).

    """
    if val < 2.07:   return 0
    elif val < 4.13: return 1
    else:            return 2


def step_3_discretization(df):
    # Keep a copy of the raw data for "Before vs After" comparison
    df_before = df.copy()
    # 3.2: Defining discretization functions 

    # 3.3: Applying transformations to the features confirmed in Table 1 and Fig. 4
    df['age']      = df['age'].apply(discretize_age)
    df['trestbps'] = df['trestbps'].apply(discretize_trestbps)
    df['chol']     = df['chol'].apply(discretize_chol)
    df['thalach']  = df['thalach'].apply(discretize_thalach)
    df['oldpeak']  = df['oldpeak'].apply(discretize_oldpeak)  

    # Rounding 'ca' and 'thal' to maintain categorical integer format after mean imputation
    df['ca']   = df['ca'].round().astype(int)
    df['thal'] = df['thal'].round().astype(int)

    print("\n--- Step 3: Discretization Process ---")
    # --- Handling the Oldpeak Contradiction ---
    print("\n[REPRODUCTION NOTE] Step 3: Oldpeak Discretization")
    print("Contradiction detected: The paper's text suggests discretization for all numeric features,")
    print("but Figure 4 clearly shows raw float values for 'oldpeak' (e.g., 3.5, 1.4).")
    print("Decision: Applying discretization to maintain consistency with the paper's methodology.\n")

    # 
    # Comparison: Before and After Discretization
    print("--- TABLE: BEFORE DISCRETIZATION (Raw Data) ---")
    print(df_before.head(10))

    print("\n--- TABLE: AFTER DISCRETIZATION (Final Reproduction Format) ---")
    print(df.head(10))
    return df

# ======================================================================================
# Step 4 Logic: NB-SKDR Pairwise Dependency (The "Difference" Method)
# ======================================================================================

def calculate_pairwise_difference(df, f1, f2, target, p_c):
    """
    Implements Equation (6) and computes the absolute difference 
    between class probabilities as described in Step 3 of the paper.
    """
    total = len(df)
    dependency_score = 0.0
    
    # Unique values for the feature pair (v1 from f1, v2 from f2)
    v1_values = df[f1].unique()
    v2_values = df[f2].unique()
    
    # Subsets for each class
    subset_c1 = df[df[target] == 1] # Diseased
    subset_c0 = df[df[target] == 0] # Healthy
    
    for v1 in v1_values:
        for v2 in v2_values:
            # 1. Marginal probabilities (Denominator of Eq 6)
            p_f1 = len(df[df[f1] == v1]) / total
            p_f2 = len(df[df[f2] == v2]) / total
            denom = p_f1 + p_f2
            
            if denom > 0:
                # 2. Probability for Class 1 (Diseased)
                p_f1_c1 = len(subset_c1[subset_c1[f1] == v1]) / len(subset_c1)
                p_f2_c1 = len(subset_c1[subset_c1[f2] == v2]) / len(subset_c1)
                prob_c1 = (p_c[1] * p_f1_c1 * p_f2_c1) / denom
                
                # 3. Probability for Class 0 (Healthy)
                p_f1_c0 = len(subset_c0[subset_c0[f1] == v1]) / len(subset_c0)
                p_f2_c0 = len(subset_c0[subset_c0[f2] == v2]) / len(subset_c0)
                prob_c0 = (p_c[0] * p_f1_c0 * p_f2_c0) / denom
                
                # 4. The "Difference" step: How well does this pair separate the classes?
                dependency_score += abs(prob_c1 - prob_c0)
                
    return dependency_score

def step_4_feature_selection(df):
    """
    Full implementation of the Feature Selection Phase (Section 3.3.2)
    """
    print("\n" + "="*60)
    print("STEP 4: NB-SKDR FEATURE SELECTION (PAIRWISE DIFFERENCE)")
    print("="*60)
    
    target = 'target'
    features = [col for col in df.columns if col != target]
    total = len(df)
    
    # Prior probability P(c)
    p_c = {c: count/total for c, count in df[target].value_counts().items()}
    feature_scores = {f: 0.0 for f in features}
    
    # Step 2 from paper: Creating pairs A={A0, A1...} and evaluating dependency
    print("Computing dependencies for all feature combinations...")
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            
            # Step 3 from paper: Equation (6) + Difference
            pair_dependency = calculate_pairwise_difference(df, f1, f2, target, p_c)
            
            # Accumulate scores for each feature in the pair
            feature_scores[f1] += pair_dependency
            feature_scores[f2] += pair_dependency

    # Sorting based on total dependency scores
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== FINAL FEATURE RANKING (NB-SKDR) ===")
    print(f"{'Rank':<5} | {'Feature':<12} | {'Total Dependency Score':<10}")
    print("-" * 50)
    for rank, (feat, score) in enumerate(ranked_features, 1):
        marker = " [TOP 6]" if rank <= 6 else ""
        print(f"{rank:<5} | {feat:<12} | {score:.4f}{marker}")
        
    # Manual selection from paper for the next phase (continuity)
    paper_features = ['age', 'sex', 'trestbps', 'fbs', 'chol', 'exang']
    
    print("\n" + "-"*60)
    print(f"Algorithm Selected Top 6: {[f for f, s in ranked_features[:6]]}")
    print(f"Paper's Final Subset B:   {paper_features}")
    print("-" * 60)
    
    return paper_features

if __name__ == "__main__":
    raw_df = step_1_load_data()
    df_cleaned = step_2_handle_missing_values(raw_df)
    df_discretized = step_3_discretization(df_cleaned)
    feature_selected = step_4_feature_selection(df_discretized)
    # classification.step_5_classification_phase(df_discretized, feature_selected)