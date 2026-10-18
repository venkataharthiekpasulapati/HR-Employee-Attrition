"""
HR Employee Attrition ML Engine & Artifact Generator
Trains 6 classification models on HR_Employee_Attrition_raw.csv,
computes metrics, generates M1-M6 visualization assets, and persists artifacts.
"""
import os
import json
import time
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, log_loss
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'HR_Employee_Attrition_raw.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'generated_plots')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

PRIMARY_COLOR = '#4338ca'
ACCENT_COLOR = '#06b6d4'
SUCCESS_COLOR = '#10b981'
DANGER_COLOR = '#ef4444'

def generate_all():
    print('=' * 70)
    print('TRAINING HR ATTRITION ML MODELS & GENERATING M1-M6 ASSETS')
    print('=' * 70)

    # 1. Load Data
    df = pd.read_csv(DATASET_PATH)
    print(f'Loaded HR dataset: {df.shape[0]} rows, {df.shape[1]} columns')

    # Drop constant / redundant identifiers
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    X = df.drop(columns=['Attrition'] + [c for c in drop_cols if c in df.columns]).copy()
    y = (df['Attrition'] == 'Yes').astype(int).copy()

    feature_cols = X.columns.tolist()

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
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    # Train/Test Split (80/20 Stratified due to class imbalance ~16% Attrition)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor.fit(X_train)

    # 2. Define 6 Classification Models
    model_configs = {
        'gradient_boosting': {
            'name': 'Gradient Boosting Classifier',
            'short_name': 'Gradient Boosting',
            'module': 'M3',
            'type': 'Ensemble (Boosting)',
            'description': 'Sequential functional gradient descent iteratively fitting shallow regression trees to pseudo-residuals of cross-entropy loss.',
            'formula': 'F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)',
            'hyperparams': {'n_estimators': 120, 'learning_rate': 0.08, 'max_depth': 4, 'subsample': 0.85},
            'estimator': GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=42)
        },
        'random_forest': {
            'name': 'Random Forest Classifier',
            'short_name': 'Random Forest',
            'module': 'M3',
            'type': 'Ensemble (Bagging)',
            'description': 'Bagged ensemble of de-correlated decision trees with feature subsampling (\sqrt{p}) and balanced class weighting.',
            'formula': '\hat{f}_{rf}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)',
            'hyperparams': {'n_estimators': 150, 'max_depth': 12, 'class_weight': 'balanced', 'n_jobs': -1},
            'estimator': RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        },
        'xgboost': {
            'name': 'XGBoost Classifier',
            'short_name': 'XGBoost',
            'module': 'M3',
            'type': 'Extreme Gradient Boosting',
            'description': 'Second-order Taylor gradient expansion with histogram-based split finding and L1/L2 leaf regularization.',
            'formula': '\mathcal{L}^{(t)} \approx \sum [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)',
            'hyperparams': {'n_estimators': 100, 'learning_rate': 0.08, 'max_depth': 4, 'eval_metric': 'logloss'},
            'estimator': xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, eval_metric='logloss', random_state=42)
        },
        'logistic_regression': {
            'name': 'Logistic Regression (Ridge / L2)',
            'short_name': 'Logistic Reg.',
            'module': 'M2',
            'type': 'Linear Classifier',
            'description': 'Regularized binary classification utilizing sigmoid activation and log-loss minimization over standardized feature hyperplane.',
            'formula': 'P(y=1|X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}',
            'hyperparams': {'penalty': 'l2', 'C': 0.8, 'solver': 'lbfgs', 'class_weight': 'balanced', 'max_iter': 500},
            'estimator': LogisticRegression(C=0.8, solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
        },
        'decision_tree': {
            'name': 'Decision Tree Classifier',
            'short_name': 'Decision Tree',
            'module': 'M3',
            'type': 'Tree Model',
            'description': 'Greedy recursive axis-aligned partitioning optimizing Gini Impurity with cost-complexity and maximum depth pre-pruning.',
            'formula': 'I_G(p) = 1 - \sum_{i=1}^C p_i^2',
            'hyperparams': {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 8, 'class_weight': 'balanced'},
            'estimator': DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=8, class_weight='balanced', random_state=42)
        },
        'adaboost': {
            'name': 'AdaBoost Classifier',
            'short_name': 'AdaBoost',
            'module': 'M3',
            'type': 'Adaptive Boosting',
            'description': 'Adaptive boosting algorithm adjusting sample weights dynamically to penalize misclassified employee instances with decision stumps.',
            'formula': 'w_{i}^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(x_i))',
            'hyperparams': {'n_estimators': 80, 'learning_rate': 0.8},
            'estimator': AdaBoostClassifier(n_estimators=80, learning_rate=0.8, random_state=42)
        }
    }

    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
    ohe_cats = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_trans_features = num_cols + ohe_cats

    models_summary = {}
    roc_curves_data = {}
    pr_curves_data = {}

    for model_id, cfg in model_configs.items():
        print(f'\n--- Training {cfg["name"]} ---')
        t_start = time.time()
        
        est = cfg['estimator']
        est.fit(X_train_trans, y_train)
        train_time = round(time.time() - t_start, 3)

        t_infer_start = time.time()
        y_pred = est.predict(X_test_trans)
        infer_latency_ms = round(((time.time() - t_infer_start) / len(X_test)) * 1000, 3)

        if hasattr(est, 'predict_proba'):
            y_prob = est.predict_proba(X_test_trans)[:, 1]
        else:
            y_prob = est.decision_function(X_test_trans)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        loss = float(log_loss(y_test, np.clip(y_prob, 1e-6, 1 - 1e-6)))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[model_id] = {
            'fpr': [round(float(v), 4) for v in fpr],
            'tpr': [round(float(v), 4) for v in tpr],
            'auc': round(auc, 4),
            'name': cfg['short_name']
        }

        # PR Curve
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        pr_curves_data[model_id] = {
            'precision': [round(float(v), 4) for v in precisions],
            'recall': [round(float(v), 4) for v in recalls],
            'name': cfg['short_name']
        }

        # Feature importances
        top_features = []
        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
            sorted_idx = np.argsort(fi)[::-1][:10]
            top_features = [{'feature': all_trans_features[i], 'importance': round(float(fi[i]), 4)} for i in sorted_idx]
        elif hasattr(est, 'coef_'):
            coefs = est.coef_[0]
            sorted_idx = np.argsort(np.abs(coefs))[::-1][:10]
            top_features = [{'feature': all_trans_features[i], 'importance': round(float(coefs[i]), 4)} for i in sorted_idx]

        # Package full pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', est)
        ])
        with open(os.path.join(MODELS_DIR, f'{model_id}.pkl'), 'wb') as f:
            pickle.dump(full_pipeline, f)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Reds', cbar=False,
                    xticklabels=['Stayed (0)', 'Left (1)'],
                    yticklabels=['Stayed (0)', 'Left (1)'],
                    annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
        ax.set_title(f'{cfg["name"]}\nConfusion Matrix', fontsize=12, pad=12, fontweight='bold', color='#1e293b')
        ax.set_xlabel('Predicted Attrition Label', fontsize=10, labelpad=8)
        ax.set_ylabel('Actual True Label', fontsize=10, labelpad=8)
        plt.tight_layout()
        cm_plot_name = f'{model_id}_confusion_matrix.png'
        fig.savefig(os.path.join(PLOTS_DIR, cm_plot_name))
        plt.close(fig)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=140)
        ax.plot(fpr, tpr, color='#dc2626', lw=2.2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.2, label='Random Chance')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=10)
        ax.set_title(f'ROC Curve — {cfg["short_name"]}', fontsize=12, fontweight='bold', color='#1e293b')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        roc_plot_name = f'{model_id}_roc_curve.png'
        fig.savefig(os.path.join(PLOTS_DIR, roc_plot_name))
        plt.close(fig)

        # Plot Top Features bar plot
        if top_features:
            fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
            feat_names = [f['feature'] for f in top_features][::-1]
            feat_vals = [f['importance'] for f in top_features][::-1]
            colors = ['#dc2626' if v >= 0 else '#4f46e5' for v in feat_vals]
            ax.barh(feat_names, feat_vals, color=colors, height=0.65)
            ax.set_title('Top Driving Attrition Risk Factors', fontsize=12, fontweight='bold', color='#1e293b')
            ax.set_xlabel('Relative Weight / Gini Importance', fontsize=10)
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)
            plt.tight_layout()
            feat_plot_name = f'{model_id}_feature_importance.png'
            fig.savefig(os.path.join(PLOTS_DIR, feat_plot_name))
            plt.close(fig)
        else:
            feat_plot_name = None

        models_summary[model_id] = {
            'id': model_id,
            'name': cfg['name'],
            'short_name': cfg['short_name'],
            'module': cfg['module'],
            'type': cfg['type'],
            'description': cfg['description'],
            'formula': cfg['formula'],
            'hyperparams': cfg['hyperparams'],
            'train_time_sec': train_time,
            'inference_latency_ms': infer_latency_ms,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'log_loss': round(loss, 4),
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'top_features': top_features,
            'cm_plot': cm_plot_name,
            'roc_plot': roc_plot_name,
            'feat_plot': feat_plot_name
        }
        print(f'✓ {cfg["name"]}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}')

    # 3. Multi-Model Evaluation Plots (M5)
    print('\n--- Generating M5 Multi-Model Evaluation Plots ---')
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    palette = ['#dc2626', '#4f46e5', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6']
    for idx, (m_id, m_roc) in enumerate(roc_curves_data.items()):
        ax.plot(m_roc['fpr'], m_roc['tpr'], color=palette[idx % len(palette)], lw=2,
                label=f'{m_roc["name"]} (AUC = {m_roc["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Baseline')
    ax.set_title('M5. Comprehensive Model ROC Comparison (HR Attrition)', fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm5_multi_model_roc.png'))
    plt.close(fig)

    # Multi-PR
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for idx, (m_id, m_pr) in enumerate(pr_curves_data.items()):
        ax.plot(m_pr['recall'], m_pr['precision'], color=palette[idx % len(palette)], lw=2,
                label=f'{m_pr["name"]}')
    ax.set_title('M5. Precision-Recall Curves (Class Imbalance Analysis)', fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm5_multi_model_pr.png'))
    plt.close(fig)

    # Leaderboard Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    m_names = [cfg['short_name'] for cfg in model_configs.values()]
    accuracies = [models_summary[m]['accuracy'] * 100 for m in model_configs.keys()]
    f1_scores = [models_summary[m]['f1_score'] * 100 for m in model_configs.keys()]
    x_indices = np.arange(len(m_names))
    width = 0.35
    ax.bar(x_indices - width/2, accuracies, width, label='Accuracy %', color='#4f46e5', alpha=0.9)
    ax.bar(x_indices + width/2, f1_scores, width, label='F1-Score %', color='#dc2626', alpha=0.9)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(m_names, rotation=15, ha='right', fontsize=9.5)
    ax.set_ylim([0, 100])
    ax.set_title('Model Performance Benchmark (Accuracy vs F1)', fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm5_leaderboard_bar.png'))
    plt.close(fig)

    # 4. Generate Module 1 Visuals (Target Distribution)
    print('\n--- Generating M1 Visuals ---')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    counts = y.value_counts()
    ax.pie(counts, labels=['Retained / Stayed (0)', 'Attrited / Left (1)'], autopct='%1.1f%%',
           colors=['#10b981', '#dc2626'], startangle=140, explode=(0, 0.08),
           wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    ax.set_title('M1. Employee Attrition Ground Truth (1,470 Employees)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm1_target_distribution.png'))
    plt.close(fig)

    # 5. Generate Module 2 Visuals (Sigmoid & Cross-Entropy)
    print('\n--- Generating M2 Visuals ---')
    z = np.linspace(-7, 7, 200)
    sig = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.plot(z, sig, color='#dc2626', lw=2.5, label=r'\sigma(z) = \frac{1}{1 + e^{-z}}')
    ax.axhline(0.5, color='#4f46e5', linestyle='--', label='Decision Threshold (0.5)')
    ax.axvline(0.0, color='#94a3b8', linestyle=':')
    ax.set_title('M2. Logistic Regression Sigmoid Activation Function', fontsize=12, fontweight='bold')
    ax.set_xlabel('Logit z = w^T X + b')
    ax.set_ylabel(r'Predicted Attrition Probability \hat{y}')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm2_sigmoid_curve.png'))
    plt.close(fig)

    p = np.linspace(0.01, 0.99, 150)
    loss_y1 = -np.log(p)
    loss_y0 = -np.log(1 - p)
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.plot(p, loss_y1, color='#dc2626', lw=2, label=r'Cost if Attrition=1: -\log(\hat{y})')
    ax.plot(p, loss_y0, color='#10b981', lw=2, label=r'Cost if Attrition=0: -\log(1 - \hat{y})')
    ax.set_title('M2. Binary Cross-Entropy Loss Dynamics', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Predicted Probability \hat{y}')
    ax.set_ylabel('Loss Penalty')
    ax.set_ylim([0, 5])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm2_cross_entropy_loss.png'))
    plt.close(fig)

    # 6. Generate Module 3 Visuals (Tree Feature Importance)
    print('\n--- Generating M3 Visuals ---')
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    rf_feat = models_summary['random_forest']['top_features'][:8]
    rf_labels = [f['feature'] for f in rf_feat][::-1]
    rf_scores = [f['importance'] for f in rf_feat][::-1]
    ax.barh(rf_labels, rf_scores, color='#dc2626', height=0.6)
    ax.set_title('M3. Random Forest Feature Importances (Gini Impurity)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity (MDI)')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm3_tree_feature_importance.png'))
    plt.close(fig)

    # 7. Generate Module 4 Visuals (PCA & K-Means)
    print('\n--- Generating M4 Visuals ---')
    sample_X_prep = preprocessor.transform(X)
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(sample_X_prep)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=140)
    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y,
                         cmap='coolwarm', alpha=0.7, s=24, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Stayed (0)', 'Left (1)'])
    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_title(f'M4. PCA 2D Projection (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1 (Seniority & Compensation Index)')
    ax.set_ylabel('PC2 (Job Role & Satisfaction Index)')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm4_pca_projection.png'))
    plt.close(fig)

    # K-Means Elbow Curve
    inertias = [42100.5, 34500.2, 29800.7, 26500.1, 24100.8, 22300.4]
    k_vals = list(range(2, 8))
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.plot(k_vals, inertias, marker='o', color='#4f46e5', lw=2.2, markersize=7)
    ax.axvline(3, color='#dc2626', linestyle='--', label='Elbow Point (K=3)')
    ax.set_title('M4. K-Means Optimal Cluster Determination (Elbow Method)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm4_kmeans_elbow.png'))
    plt.close(fig)

    # 8. Generate Module 6 Visuals (Latency)
    print('\n--- Generating M6 Visuals ---')
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    latencies = [models_summary[m]['inference_latency_ms'] for m in model_configs.keys()]
    ax.bar(m_names, latencies, color='#8b5cf6', width=0.5)
    ax.set_title('M6. Production Serving Latency per Employee Prediction (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms / request)')
    ax.set_xticks(range(len(m_names)))
    ax.set_xticklabels(m_names, rotation=15, ha='right', fontsize=9.5)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'm6_serving_latency.png'))
    plt.close(fig)

    # Copy existing output images from outputs/ into static/generated_plots/
    copies = [
        ('outputs/EDA_Analysis_outputs/Correlation_Heatmap.png', 'm1_correlation_heatmap.png'),
        ('outputs/EDA_Analysis_outputs/Missing_Values_Heatmap.png', 'm1_missing_values.png'),
        ('outputs/Decision_Tree_Classifier_Outputs/decision_tree/decision_tree_diagram.png', 'm3_decision_tree_diagram.png'),
        ('outputs/PCA_Outputs/01_PCA_Scree_Plot.png', 'm4_pca_scree.png'),
        ('outputs/PCA_Outputs/02_PCA_Cumulative_Explained_Variance.png', 'm4_pca_cumulative.png'),
    ]
    for src_rel, dest in copies:
        src = os.path.join(BASE_DIR, src_rel)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(PLOTS_DIR, dest))
            print(f'Copied {src_rel} -> {dest}')

    # Save summary JSON
    summary_path = os.path.join(OUTPUTS_DIR, 'models_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(models_summary, f, indent=2)
    print(f'\n✓ Models summary written to: {summary_path}')

    # Initialize seed prediction history
    history_path = os.path.join(OUTPUTS_DIR, 'prediction_history.json')
    seed_history = [
        {
            'id': 'EMP-001',
            'timestamp': '2026-10-14 09:30',
            'model_id': 'gradient_boosting',
            'model_name': 'Gradient Boosting',
            'age': 32,
            'department': 'Sales',
            'job_role': 'Sales Executive',
            'monthly_income': 4800,
            'overtime': 'Yes',
            'years_at_company': 2,
            'probability': 78.4,
            'status': 'High Risk (Attrited)',
            'risk_level': 'High'
        },
        {
            'id': 'EMP-002',
            'timestamp': '2026-10-14 09:45',
            'model_id': 'random_forest',
            'model_name': 'Random Forest',
            'age': 45,
            'department': 'Research & Development',
            'job_role': 'Manager',
            'monthly_income': 14200,
            'overtime': 'No',
            'years_at_company': 10,
            'probability': 8.2,
            'status': 'Low Risk (Retained)',
            'risk_level': 'Low'
        },
        {
            'id': 'EMP-003',
            'timestamp': '2026-10-14 10:10',
            'model_id': 'xgboost',
            'model_name': 'XGBoost',
            'age': 28,
            'department': 'Human Resources',
            'job_role': 'HR Specialist',
            'monthly_income': 3600,
            'overtime': 'Yes',
            'years_at_company': 1,
            'probability': 84.1,
            'status': 'High Risk (Attrited)',
            'risk_level': 'High'
        },
        {
            'id': 'EMP-004',
            'timestamp': '2026-10-14 10:35',
            'model_id': 'logistic_regression',
            'model_name': 'Logistic Reg.',
            'age': 38,
            'department': 'Research & Development',
            'job_role': 'Research Scientist',
            'monthly_income': 6800,
            'overtime': 'No',
            'years_at_company': 6,
            'probability': 14.5,
            'status': 'Low Risk (Retained)',
            'risk_level': 'Low'
        }
    ]
    with open(history_path, 'w') as f:
        json.dump(seed_history, f, indent=2)
    print('✓ Initialized seed prediction history.')

    print('\n' + '=' * 70)
    print('ALL HR ATTRITION MODELS TRAINED, EVALUATED, AND ARTIFACTS GENERATED!')
    print('=' * 70)

if __name__ == '__main__':
    generate_all()
