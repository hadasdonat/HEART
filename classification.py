import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from main import df, selected_features

def evaluate_classifiers(df, features, label):
    X = df[features]
    y = df['target']

    # חלוקה מדויקת לפי המאמר: 106 דגימות testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=106, random_state=42, stratify=y
    )

    print(f"\n=== {label} ===")
    print(f"Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    print(f"\n{'Algorithm':<12}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}")
    print("-" * 70)

    classifiers = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'DT' : DecisionTreeClassifier(criterion='entropy', random_state=42),
        'RF' : RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

        accuracy    = (TP+TN) / (TP+TN+FP+FN) * 100
        sensitivity = TP / (TP+FN) * 100
        specificity = TN / (TN+FP) * 100

        print(f"{name:<12}{TP:>5}{TN:>5}{FP:>5}{FN:>5}{accuracy:>9.1f}%{sensitivity:>12.1f}%{specificity:>12.1f}%")

# Experiment 1: all 13 features
all_features = [col for col in df.columns if col != 'target']
evaluate_classifiers(df, all_features, "Experiment 1: 13 Features (Without Feature Selection)")

# Experiment 2: 6 selected features (as per paper)
evaluate_classifiers(df, selected_features, "Experiment 2: 6 Features (With Feature Selection)")