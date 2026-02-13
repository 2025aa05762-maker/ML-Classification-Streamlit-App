import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Load dataset
data = pd.read_csv("../data/breast-cancer.csv")

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

y = y.map({'M':1, 'B':0})
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    results.append([name, accuracy, auc, precision, recall, f1, mcc])

    joblib.dump(model, f"{name}.pkl")

result_df = pd.DataFrame(
    results,
    columns=["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]
)

print(result_df)

# save model comparison table
result_df.to_csv("../model_comparison.csv", index=False)

# save sample test data for streamlit upload
test_sample = X_test.copy()
test_sample["target"] = y_test
test_sample.to_csv("../testdata.csv", index=False)
