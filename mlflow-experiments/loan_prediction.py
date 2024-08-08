import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
from sklearn import metrics

# Dataset
dataset_path = os.path.abspath("./datasets")
train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))

# Filtering
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# Filling blanks
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in numerical_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Filling Outliers
train_df[numerical_cols] = train_df[numerical_cols].apply(lambda x:x.clip(*x.quantile([0.05, 0.95])))

# Normalization
train_df['LoanAmount'] = np.log(train_df['LoanAmount']).copy()
train_df['TotalIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
train_df['TotalIncome'] = np.log(train_df['TotalIncome']).copy()

# Dropping
train_df = train_df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])

train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])

# Train Test Split
X = train_df.drop(columns=['Loan_Status', 'Loan_ID'])
y = train_df['Loan_Status']
RANDOM_SEED = 6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# Models
## Random Forests
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200, 400, 700],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

modelforest = grid_forest.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['liblinear']
}

grid_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_lr = grid_lr.fit(X_train, y_train)

# Decision Trees
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_dt = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'criterion': ['gini', 'entropy']
}

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_dt = grid_dt.fit(X_train, y_train)

# ML Flow

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve area: {auc:0.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/ROC_Curve.png")
    plt.close()
    return accuracy, f1, auc

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        pred = model.predict(X)

        #metrics
        accuracy, f1, auc = eval_metrics(y, pred)

        mlflow.log_params(model.best_params_)

        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('F1-score', f1)
        mlflow.log_metric("AUC", auc)

        mlflow.log_artifact('plots/ROC_Curve.png')
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()

mlflow.set_experiment('Loan_Prediction')
mlflow_logging(model=modelforest, X=X_test, y=y_test, name="RandomForestClassifier")
mlflow_logging(model=model_lr, X=X_test, y=y_test, name="LogisticRegression")
mlflow_logging(model=model_dt, X=X_test, y=y_test, name="DecisionTree")
print("Done.")