import pandas as pd

def search_image_data():
    # טעינת הנתונים הגולמיים
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=column_names, na_values='?')

    # הגדרת 5 השורות הראשונות כפי שהן מופיעות באיור 1 (Fig. 1) ובאיור 4 (Fig. 4)
    # שים לב: העתקתי את הערכים בדיוק מהתמונות ששלחת
    rows_from_images = [
        {"id": "Row 1", "data": {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "thalach": 150, "oldpeak": 2.3, "target": 1}},
        {"id": "Row 2", "data": {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "thalach": 108, "oldpeak": 1.5, "target": 1}},
        {"id": "Row 3", "data": {"age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 229, "thalach": 129, "oldpeak": 2.6, "target": 1}},
        {"id": "Row 4", "data": {"age": 37, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250, "thalach": 187, "oldpeak": 3.5, "target": 1}},
        {"id": "Row 5", "data": {"age": 41, "sex": 0, "cp": 1, "trestbps": 130, "chol": 204, "thalach": 172, "oldpeak": 1.4, "target": 1}}
    ]

    print(f"{'Row ID':<10} | {'Exact Match':<12} | {'Patient Match':<15} | {'Discrepancy'}")
    print("-" * 80)

    for row in rows_from_images:
        d = row["data"]
        
        # 1. חיפוש מדויק (הכל חייב להתאים)
        exact = df[(df['age'] == d['age']) & (df['trestbps'] == d['trestbps']) & 
                   (df['chol'] == d['chol']) & (df['cp'] == d['cp']) & (df['target'] == d['target'])]
        
        # 2. חיפוש לפי פרופיל רפואי (גיל, לחץ דם, כולסטרול, דופק) - לזיהוי המטופל המקורי
        patient = df[(df['age'] == d['age']) & (df['trestbps'] == d['trestbps']) & 
                     (df['chol'] == d['chol']) & (df['thalach'] == d['thalach'])]

        exact_str = "✅ YES" if not exact.empty else "❌ NO"
        patient_str = f"✅ Index {patient.index[0]}" if not patient.empty else "❌ NOT FOUND"
        
        # זיהוי מה השתנה
        diff = ""
        if exact.empty and not patient.empty:
            orig = patient.iloc[0]
            if orig['target'] != d['target']:
                diff += f"Target changed ({orig['target']}->{d['target']}) "
            if orig['cp'] != d['cp']:
                diff += f"CP changed ({orig['cp']}->{d['cp']})"

        print(f"{row['id']:<10} | {exact_str:<12} | {patient_str:<15} | {diff}")

if __name__ == "__main__":
    search_image_data()