import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ======================================================================================
# STEP 6: DATA PREPARATION
# ======================================================================================

def prepare_experiment_data(df, features):
    """
    6.1 Remove features not in the selected list.
    6.2 Ensure all values are Integers for discrete logic.
    """
    print(f"[Audit - Step 6] Isolating {len(features)} features...")
    
    # Selecting the features and the target
    X = df[features].copy().astype(int)
    y = df['target'].copy().astype(int)
    
    print(f"      - Data shape: {X.shape}")
    print(f"      - Sample features: {list(X.columns)}")
    return X, y

# ======================================================================================
# STEP 7: CLASSIFICATION AND TRAINING
# ======================================================================================

def partition_and_train_models(X, y):
    """
    7.1 Split data (65% Training / 35% Testing).
    7.2 Set up algorithms: NB, SVM, DT, RF, KNN.
    7.3/7.4 Train and Generate Predictions.
    """
    # 7.1: Data Partitioning (35% test = 106 samples per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=106, random_state=42, stratify=y
    )
    
    print(f"[Audit - Step 7.1] Split completed. Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[Audit - Step 7.1] Class distribution in Test set: {y_test.value_counts().to_dict()}")

    # 7.2: Refined Model Definitions for better reproduction
    models = {
        'Naive Bayes': CategoricalNB(),
        'SVM': SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42),
       'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
       'Decision Tree': DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=4,           # הגבלה לעומק רדוד יותר
            min_samples_leaf=10,    # הגדלת כמות הדגימות המינימלית בעלה
            random_state=42
        ), 
        'KNN': KNeighborsClassifier(
                    n_neighbors=9,         # העלאת מספר השכנים
                    metric='manhattan',    # שימוש במרחק אוקלידי
                    weights='distance'     # מתן משקל לפי מרחק
        )
                }

    predictions = {}
    
    print(f"[Audit - Step 7.3/4] Training models and generating predictions...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)
        
    return y_test, predictions

# ======================================================================================
# STEP 8: EVALUATION AND METRICS
# ======================================================================================

def calculate_performance_metrics(y_test, predictions_dict, experiment_label):
    """
    8.1/8.2 Calculate TP, TN, FP, FN and evaluation formulas.
    8.3 Print final comparison table.
    """
    print("\n" + "="*85)
    print(f"RESULTS: {experiment_label.upper()}")
    print("="*85)
    print(f"{'Algorithm':<15}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}")
    print("-" * 85)

    for name, y_pred in predictions_dict.items():
        # 8.1: Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # 8.2: Accuracy (Eq. 7), Sensitivity (Eq. 8), Specificity (Eq. 9)
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0

        print(f"{name:<15}{tp:>5}{tn:>5}{fp:>5}{fn:>5}{accuracy:>9.2f}%{sensitivity:>12.2f}%{specificity:>12.2f}%")

# ======================================================================================
# ORCHESTRATION FUNCTION (The function that runs the experiment)
# ======================================================================================

def run_reproduction_experiment(df, features, label):
    """
    Wraps steps 6, 7, and 8 for a single experiment.
    """
    # Step 6
    X, y = prepare_experiment_data(df, features)
    
    # Step 7
    y_test, predictions = partition_and_train_models(X, y)
    
    # Step 8
    calculate_performance_metrics(y_test, predictions, label)

def run_all_comparisons(df, selected_features):
    """
    9. Step 9: Compares Experiment 1 (All features) vs Experiment 2 (Subset B).
    """
    print("\n" + "#"*85)
    print("PHASE 9: COMPARATIVE ANALYSIS (EXPERIMENT 1 vs EXPERIMENT 2)")
    print("#"*85)

    # 9.1: Experiment 1 - All 13 Features
    all_features = [col for col in df.columns if col != 'target']
    run_reproduction_experiment(df, all_features, "Experiment 1: All 13 Features")

    # 9.2: Experiment 2 - Selected 6 Features
    run_reproduction_experiment(df, selected_features, "Experiment 2: Selected 6 Features")