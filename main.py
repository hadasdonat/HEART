import pandas as pd

# טעינת מסד הנתונים Cleveland Heart Disease מ-UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=column_names, na_values='?')

# המרת משתנה המטרה: כל ערך > 0 הופך ל-1 (יש מחלה)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"גודל הדאטה: {df.shape}")
print(f"חולים: {df['target'].sum()}, בריאים: {(df['target']==0).sum()}")
print(df.head())