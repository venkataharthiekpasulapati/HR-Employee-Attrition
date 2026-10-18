"""
MLflow Experiment Tracking & Model Registry Integration for HR Employee Attrition
Demonstrates how HR Attrition Predict integrates with MLflow (Module M6)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'HR_Employee_Attrition_raw.csv')
MLRUNS_DIR = os.path.join(BASE_DIR, 'mlruns')

def run_mlflow_experiment():
    print('=' * 65)
    print('MLflow Tracking for HR Employee Attrition (Module M6)')
    print('=' * 65)

    # 1. Set Tracking URI explicitly to project folder
    mlflow.set_tracking_uri(f'file://{MLRUNS_DIR}')

    # 2. Set MLflow Experiment
    mlflow.set_experiment('HR_Employee_Attrition_Benchmark')

    # 3. Load Data
    df = pd.read_csv(DATASET_PATH)
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    X = df.drop(columns=['Attrition'] + [c for c in drop_cols if c in df.columns]).copy()
    y = (df['Attrition'] == 'Yes').astype(int).copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Start MLflow Run
    with mlflow.start_run(run_name='Gradient_Boosting_Champion'):
        # Log Hyperparameters
        params = {
            'n_estimators': 120,
            'learning_rate': 0.08,
            'max_depth': 4,
            'subsample': 0.85,
            'test_split_size': 0.20,
            'random_state': 42
        }
        mlflow.log_params(params)

        # Train Pipeline
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            random_state=params['random_state']
        )
        pipeline = Pipeline([('prep', preprocessor), ('model', clf)])
        pipeline.fit(X_train, y_train)

        # Evaluate
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, probs))

        # Log Metrics
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', auc)

        # Log Artifacts (Plots)
        roc_plot = os.path.join(BASE_DIR, 'static', 'generated_plots', 'gradient_boosting_roc_curve.png')
        if os.path.exists(roc_plot):
            mlflow.log_artifact(roc_plot, artifact_path='evaluation_plots')

        cm_plot = os.path.join(BASE_DIR, 'static', 'generated_plots', 'gradient_boosting_confusion_matrix.png')
        if os.path.exists(cm_plot):
            mlflow.log_artifact(cm_plot, artifact_path='evaluation_plots')

        fi_plot = os.path.join(BASE_DIR, 'static', 'generated_plots', 'gradient_boosting_feature_importance.png')
        if os.path.exists(fi_plot):
            mlflow.log_artifact(fi_plot, artifact_path='evaluation_plots')

        # Log & Register the Model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name='model',
            serialization_format='pickle',
            registered_model_name='HR_Attrition_GradientBoosting'
        )

        print(f'✓ Run complete! Logged Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}')
        print(f'Tracking URI set to: {MLRUNS_DIR}')
        print('To launch the interactive MLflow UI for HR, run: mlflow ui --port 5003')

if __name__ == '__main__':
    run_mlflow_experiment()
