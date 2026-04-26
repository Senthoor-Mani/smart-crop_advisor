import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

DATA_PATH = "data/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

X = df[["N","P","K","temperature","humidity","ph","rainfall"]].copy()
y_text = df["label"].astype(str).copy()

# ✅ Encode string labels -> integers 0..K-1
le = LabelEncoder()
y = le.fit_transform(y_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
print(classification_report(y_test, pred, target_names=le.classes_))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "/content/smart-crop-advisor/models/crop_model.joblib")
joblib.dump(le, "/content/smart-crop-advisor/models/label_encoder.joblib")
print("Saved models/crop_model.joblib and models/label_encoder.joblib")
