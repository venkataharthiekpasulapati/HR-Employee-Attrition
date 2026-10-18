"""
ML Engine Service for HR Employee Attrition Prediction System
Provides model inference, attrition risk profiling, factor breakdown, dataset profiling, and dashboard analytics.
"""
import os
import json
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'HR_Employee_Attrition_raw.csv')
UPLOAD_DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'uploaded_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
SUMMARY_FILE = os.path.join(OUTPUTS_DIR, 'models_summary.json')
HISTORY_FILE = os.path.join(OUTPUTS_DIR, 'prediction_history.json')

_LOADED_MODELS = {}

def load_classification_models():
    """Returns only the 6 trained binary attrition classifier pipelines/summaries."""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_complete_models_catalog():
    """
    Returns the comprehensive catalog of ALL 17 machine learning models,
    manifold learning methods, clustering algorithms, regularizers,
    and cross-validation frameworks implemented across the project,
    arranged STRICTLY in syllabus module sequence (M2 -> M3 -> M4 -> M5).
    """
    import collections
    base_summary = load_classification_models()

    # -------------------------------------------------------------
    # MODULE 2: SUPERVISED LEARNING — LINEAR MODELS & REGULARIZATION
    # Sequence: 2.1 CFNE/GD -> 2.2 Metrics -> 2.3 Regularization -> 2.4 Logistic -> 2.5 Multinomial
    # -------------------------------------------------------------
    m2_cfne_gd = {
        'id': 'linear_regression_cfne_gd',
        'seq_id': 'Model 2.1',
        'seq_num': '2.1',
        'name': 'Linear Regression (CFNE vs Gradient Descent)',
        'short_name': 'Linear Reg. (CFNE vs GD)',
        'category': 'linear_regularization',
        'category_name': 'Linear Models & Regularization',
        'module': 'M2',
        'type': 'Analytical vs Iterative Continuous Optimization',
        'description': 'Compares analytical exact Closed-Form Normal Equation (CFNE) solutions against iterative batch Gradient Descent (GD) convergence rates and residual distributions for compensation modeling.',
        'formula': r'\text{Normal Eq: } \theta = (X^T X)^{-1} X^T y \quad \text{vs} \quad \text{GD: } \theta := \theta - \alpha \nabla_\theta J(\theta)',
        'key_metric_label': 'Convergence',
        'key_metric_val': 'Exact vs GD Iterative',
        'hyperparams': {
            'learning_rate': 0.01,
            'iterations': 1000,
            'loss_function': 'Mean Squared Error (MSE)',
            'features': 'Age, TotalWorkingYears, Education, JobLevel'
        },
        'outputs_folder': 'outputs/Linear_Regression_CFNE_GD_Compare',
        'source_script': 'src/Linear_Regression_CFNE_GD_Compare.py',
        'visual_plots': [
            {'title': 'Actual vs Predicted Continuous Values', 'url': '/outputs/Linear_Regression_CFNE_GD_Compare/actual_vs_predicted.png', 'desc': 'Regression fit comparison against ground truth targets.'},
            {'title': 'Gradient Descent Loss Curve', 'url': '/outputs/Linear_Regression_CFNE_GD_Compare/gradient_descent_loss.png', 'desc': 'MSE loss decay per iteration demonstrating convex convergence.'},
            {'title': 'Residual Error Comparison', 'url': '/outputs/Linear_Regression_CFNE_GD_Compare/residual_comparison.png', 'desc': 'Error residual distribution comparing CFNE and Gradient Descent.'}
        ],
        'code_snippet': """import numpy as np

# Closed-Form Normal Equation
theta_cfne = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
# Gradient Descent updates
theta_gd = np.zeros(X_b.shape[1])
for _ in range(1000):
    theta_gd -= 0.01 * (2 / len(y)) * X_b.T @ (X_b @ theta_gd - y)"""
    }

    m2_metrics = {
        'id': 'linear_regression_metrics',
        'seq_id': 'Model 2.2',
        'seq_num': '2.2',
        'name': 'Linear Regression with Performance Metrics',
        'short_name': 'Linear Regression Metrics',
        'category': 'linear_regularization',
        'category_name': 'Linear Models & Regularization',
        'module': 'M2',
        'type': 'Continuous Target Regression with Statistical Metrics',
        'description': 'Continuous evaluation on employee compensation modeling with comprehensive regression goodness-of-fit metrics: R-squared (R2), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).',
        'formula': r'R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}, \quad \text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}',
        'key_metric_label': 'Goodness of Fit',
        'key_metric_val': 'R² = 0.884',
        'hyperparams': {
            'fit_intercept': True,
            'metrics_evaluated': 'R2, MSE, RMSE, MAE',
            'target': 'MonthlyIncome ($)'
        },
        'outputs_folder': 'outputs/Linear_Regression_with_Metrics',
        'source_script': 'src/Linear_Regression_with_Metrics.py',
        'visual_plots': [
            {'title': 'Linear Regression Line Fit', 'url': '/outputs/Linear_Regression_with_Metrics/images/linear_regression_line.png', 'desc': 'Best-fit regression hyperplane.'},
            {'title': 'Actual vs Predicted Targets', 'url': '/outputs/Linear_Regression_with_Metrics/images/actual_vs_predicted.png', 'desc': 'Predicted vs ground truth compensation.'},
            {'title': 'Residual Error Distribution', 'url': '/outputs/Linear_Regression_with_Metrics/images/residual_distribution.png', 'desc': 'Homoscedasticity and residual symmetry analysis.'}
        ],
        'code_snippet': """from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"R2: {r2_score(y_test, y_pred):.3f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")"""
    }

    m2_regularization = {
        'id': 'regularization_ridge_lasso_elasticnet',
        'seq_id': 'Model 2.3',
        'seq_num': '2.3',
        'name': 'Regularization (Ridge L2, Lasso L1 & ElasticNet)',
        'short_name': 'Ridge / Lasso / ElasticNet',
        'category': 'linear_regularization',
        'category_name': 'Linear Models & Regularization',
        'module': 'M2',
        'type': 'Shrinkage & Feature Sparsity Penalties',
        'description': 'Systematic regularization comparison on employee attrition: L2 Ridge shrinkage preventing multicollinearity, L1 Lasso diamond constraint enforcing exact zero sparsity feature selection, and ElasticNet convex combination.',
        'formula': r'J_{\text{Elastic}}(\theta) = \text{MSE}(\theta) + r \alpha ||\theta||_1 + \frac{1 - r}{2} \alpha ||\theta||_2^2',
        'key_metric_label': 'Top Regularized Accuracy',
        'key_metric_val': '87.76% (Ridge L2)',
        'hyperparams': {
            'ridge_alpha': 1.0,
            'lasso_alpha': 0.05,
            'elastic_l1_ratio': 0.5,
            'best_model': 'Ridge L2 (87.76% Accuracy, 100% Precision on Attrition)'
        },
        'outputs_folder': 'outputs/Regularization_ridg_laso_elastic_outputs',
        'source_script': 'src/Regularizatio_ridge_laso_elastic.py',
        'visual_plots': [
            {'title': 'Regularization Model Comparison', 'url': '/outputs/Regularization_ridg_laso_elastic_outputs/Model_Performance.png', 'desc': 'Accuracy, Precision, Recall, and F1 across Ridge, Lasso, and ElasticNet.'},
            {'title': 'L1 vs L2 Constraint Geometry', 'url': '/outputs/Regularization_ridg_laso_elastic_outputs/L1_vs_L2_Regularisation.png', 'desc': 'L1 diamond corners inducing exact zero coefficients vs L2 circular shrinkage.'},
            {'title': 'Lasso Sparsity Graph', 'url': '/outputs/Regularization_ridg_laso_elastic_outputs/Lasso_Sparsity_Graph.png', 'desc': 'Feature elimination as regularization penalty alpha increases.'},
            {'title': 'Coefficient Shrinkage Comparison', 'url': '/outputs/Regularization_ridg_laso_elastic_outputs/Coefficient_Comparison.png', 'desc': 'Direct comparison of parameter magnitudes across methods.'}
        ],
        'code_snippet': """from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.05).fit(X_train, y_train)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
print(f"Lasso eliminated {sum(lasso.coef_ == 0)} redundant features")"""
    }

    lr_raw = base_summary.get('logistic_regression', {})
    m2_logistic = dict(lr_raw)
    m2_logistic.update({
        'id': 'logistic_regression',
        'seq_id': 'Model 2.4',
        'seq_num': '2.4',
        'module': 'M2',
        'name': 'Logistic Regression (Ridge / L2)',
        'short_name': 'Logistic Regression',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{lr_raw.get('roc_auc', 0.812):.4f}",
        'outputs_folder': 'outputs/Logistic_Regression_Binary_Classify',
        'source_script': 'src/Logistic_Regression_Binary_Classifi.py',
        'visual_plots': [
            {'title': 'Logistic Regression Confusion Matrix', 'url': '/static/generated_plots/logistic_regression_confusion_matrix.png', 'desc': 'Test set confusion matrix for binary attrition classification.'},
            {'title': 'Decision Boundary Hyperplane', 'url': '/outputs/Logistic_Regression_Binary_Classify/04_decision_boundary_hyperplane.png', 'desc': '2D projection of optimal separating hyperplane.'},
            {'title': 'Sigmoid Activation Function Curve', 'url': '/static/generated_plots/m2_sigmoid_curve.png', 'desc': 'Logit to probability transformation mapping z to [0, 1].'},
            {'title': 'Binary Cross-Entropy Loss Dynamics', 'url': '/static/generated_plots/m2_cross_entropy_loss.png', 'desc': 'Log-loss curvature penalizing confident misclassifications.'},
            {'title': 'Logistic Regression ROC Curve', 'url': '/static/generated_plots/logistic_regression_roc_curve.png', 'desc': 'ROC discrimination trade-off curve.'}
        ],
        'code_snippet': """from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
log_reg.fit(X_train, y_train)
probs = log_reg.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")"""
    })

    m2_multinomial = {
        'id': 'multinomial_logistic_regression',
        'seq_id': 'Model 2.5',
        'seq_num': '2.5',
        'name': 'Multinomial Logistic Regression (Multiclass Softmax)',
        'short_name': 'Multinomial Softmax',
        'category': 'linear_regularization',
        'category_name': 'Linear Models & Regularization',
        'module': 'M2',
        'type': 'Multiclass Generalized Linear Model',
        'description': 'Generalizes binary logistic regression to multiclass employee Department allocation (Research & Development, Sales, Human Resources) using Softmax probabilities and multi-class categorical cross-entropy loss.',
        'formula': r'P(Y = k \mid X) = \frac{e^{w_k^T X + b_k}}{\sum_{j=1}^K e^{w_j^T X + b_j}}, \quad \mathcal{L} = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}',
        'key_metric_label': 'Target Classes',
        'key_metric_val': '3 Departments (R&D, Sales, HR)',
        'hyperparams': {
            'multi_class': 'multinomial',
            'solver': 'lbfgs',
            'penalty': 'l2',
            'classes': 'Research & Development, Sales, Human Resources'
        },
        'outputs_folder': 'outputs/Multinomial_Logistic_Regre_for_Multiclass',
        'source_script': 'src/Multinomial_Logistic_Regre_for_Multiclas.py',
        'visual_plots': [
            {'title': 'Multiclass Confusion Matrix', 'url': '/outputs/Multinomial_Logistic_Regre_for_Multiclass/confusion_matrix.png', 'desc': '3x3 confusion matrix across employee departments.'},
            {'title': 'Department Class Distribution', 'url': '/outputs/Multinomial_Logistic_Regre_for_Multiclass/class_distribution.png', 'desc': 'Ground truth distribution across the 3 target departments.'}
        ],
        'code_snippet': """from sklearn.linear_model import LogisticRegression

multi_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_lr.fit(X_train, y_dept)
probs = multi_lr.predict_proba(X_test[:1])
print(f"Softmax Department Probabilities: {probs}")"""
    }

    m2_models = [m2_cfne_gd, m2_metrics, m2_regularization, m2_logistic, m2_multinomial]

    # -------------------------------------------------------------
    # MODULE 3: SUPERVISED LEARNING — TREE-BASED ENSEMBLES & CLASSIFIERS
    # Sequence: 3.1 Decision Tree -> 3.2 Random Forest -> 3.3 AdaBoost -> 3.4 Gradient Boosting -> 3.5 XGBoost
    # -------------------------------------------------------------
    dt_raw = base_summary.get('decision_tree', {})
    m3_decision_tree = dict(dt_raw)
    m3_decision_tree.update({
        'id': 'decision_tree',
        'seq_id': 'Model 3.1',
        'seq_num': '3.1',
        'module': 'M3',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{dt_raw.get('roc_auc', 0.65):.4f}",
        'outputs_folder': 'outputs/Decision_Tree_Classifier_Outputs',
        'source_script': 'src/Decision_Tree_Classifier.py',
        'visual_plots': [
            {'title': 'Decision Tree Pruned Architecture', 'url': '/static/generated_plots/m3_decision_tree_diagram.png', 'desc': 'Greedy axis-aligned splits optimizing Gini Impurity.'},
            {'title': 'Decision Tree Confusion Matrix', 'url': '/static/generated_plots/decision_tree_confusion_matrix.png', 'desc': 'Holdout evaluation matrix.'},
            {'title': 'Decision Tree Feature Importance', 'url': '/static/generated_plots/decision_tree_feature_importance.png', 'desc': 'Gini impurity decrease ranking.'},
            {'title': 'Decision Tree ROC Curve', 'url': '/static/generated_plots/decision_tree_roc_curve.png', 'desc': 'Single-tree discrimination performance.'},
            {'title': 'Actual vs Predicted Outcomes', 'url': '/outputs/Decision_Tree_Classifier_Outputs/charts/actual_vs_predicted.png', 'desc': 'Validation prediction distribution against ground truth.'}
        ]
    })

    rf_raw = base_summary.get('random_forest', {})
    m3_random_forest = dict(rf_raw)
    m3_random_forest.update({
        'id': 'random_forest',
        'seq_id': 'Model 3.2',
        'seq_num': '3.2',
        'module': 'M3',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{rf_raw.get('roc_auc', 0.81):.4f}",
        'outputs_folder': 'outputs/Random_Forest_Tree_Outputs',
        'source_script': 'src/Random_Forest_Tree.py',
        'visual_plots': [
            {'title': 'Random Forest Confusion Matrix', 'url': '/static/generated_plots/random_forest_confusion_matrix.png', 'desc': '150-tree bagging ensemble holdout matrix.'},
            {'title': 'MDI Feature Importances (150 Trees)', 'url': '/static/generated_plots/m3_tree_feature_importance.png', 'desc': 'Mean Decrease in Impurity across ensemble estimators.'},
            {'title': 'Random Forest ROC Curve', 'url': '/static/generated_plots/random_forest_roc_curve.png', 'desc': 'Ensemble ROC discrimination curve.'},
            {'title': 'Individual Tree Split Diagram', 'url': '/outputs/Random_Forest_Tree_Outputs/random_forest_tree/random_forest_tree_1.png', 'desc': 'Architecture of single de-correlated estimator tree #1.'},
            {'title': 'Actual vs Predicted Outcomes', 'url': '/outputs/Random_Forest_Tree_Outputs/charts/actual_vs_predicted.png', 'desc': 'Holdout comparison against ground truth.'}
        ]
    })

    ada_raw = base_summary.get('adaboost', {})
    m3_adaboost = dict(ada_raw)
    m3_adaboost.update({
        'id': 'adaboost',
        'seq_id': 'Model 3.3',
        'seq_num': '3.3',
        'module': 'M3',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{ada_raw.get('roc_auc', 0.80):.4f}",
        'outputs_folder': 'outputs/AdaBoost_Classifier_Outputs',
        'source_script': 'src/AdaBoost_Classifier.py',
        'visual_plots': [
            {'title': 'AdaBoost Confusion Matrix', 'url': '/static/generated_plots/adaboost_confusion_matrix.png', 'desc': 'Sequential error-weighted confusion matrix.'},
            {'title': 'AdaBoost Feature Importances', 'url': '/static/generated_plots/adaboost_feature_importance.png', 'desc': 'Weight contributions across sequential boosting stumps.'},
            {'title': 'AdaBoost ROC Curve', 'url': '/static/generated_plots/adaboost_roc_curve.png', 'desc': 'Adaptive boosting ROC curve.'},
            {'title': 'Base Estimator Decision Stump', 'url': '/outputs/AdaBoost_Classifier_Outputs/adaboost_tree/adaboost_base_tree_1.png', 'desc': 'Weak learner architecture diagram for stump #1.'},
            {'title': 'Actual vs Predicted Outcomes', 'url': '/outputs/AdaBoost_Classifier_Outputs/charts/actual_vs_predicted.png', 'desc': 'AdaBoost prediction alignment.'}
        ]
    })

    gb_raw = base_summary.get('gradient_boosting', {})
    m3_gradient_boosting = dict(gb_raw)
    m3_gradient_boosting.update({
        'id': 'gradient_boosting',
        'seq_id': 'Model 3.4',
        'seq_num': '3.4',
        'module': 'M3',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{gb_raw.get('roc_auc', 0.825):.4f}",
        'outputs_folder': 'outputs/GBoost_Classifier_Outputs',
        'source_script': 'src/GB_Boost_Classifier.py',
        'visual_plots': [
            {'title': 'Gradient Boosting Confusion Matrix', 'url': '/static/generated_plots/gradient_boosting_confusion_matrix.png', 'desc': 'Functional gradient holdout evaluation.'},
            {'title': 'Gradient Boosting ROC Curve', 'url': '/static/generated_plots/gradient_boosting_roc_curve.png', 'desc': 'Discrimination trade-off under residual descent.'},
            {'title': 'Pseudo-Residual Feature Importance', 'url': '/static/generated_plots/gradient_boosting_feature_importance.png', 'desc': 'Relative importance in minimizing log-loss residuals.'}
        ]
    })

    xgb_raw = base_summary.get('xgboost', {})
    m3_xgboost = dict(xgb_raw)
    m3_xgboost.update({
        'id': 'xgboost',
        'seq_id': 'Model 3.5',
        'seq_num': '3.5',
        'module': 'M3',
        'category': 'classification',
        'category_name': 'Supervised Classification',
        'key_metric_label': 'ROC-AUC',
        'key_metric_val': f"{xgb_raw.get('roc_auc', 0.825):.4f}",
        'outputs_folder': 'outputs/XGBoost_Classifier_Outputs',
        'source_script': 'src/XGBoost_Classifier.py',
        'visual_plots': [
            {'title': 'XGBoost Confusion Matrix', 'url': '/static/generated_plots/xgboost_confusion_matrix.png', 'desc': 'Top classification test set confusion matrix.'},
            {'title': 'XGBoost ROC Curve (Top AUC: 0.825)', 'url': '/static/generated_plots/xgboost_roc_curve.png', 'desc': 'Discrimination curve with second-order Taylor expansion optimization.'},
            {'title': 'XGBoost Feature Importance (Gain)', 'url': '/static/generated_plots/xgboost_feature_importance.png', 'desc': 'Regularized split gain importance ranking.'}
        ]
    })

    m3_models = [m3_decision_tree, m3_random_forest, m3_adaboost, m3_gradient_boosting, m3_xgboost]

    # -------------------------------------------------------------
    # MODULE 4: UNSUPERVISED LEARNING, CLUSTERING & DIMENSIONALITY
    # Sequence: 4.1 PCA -> 4.2 t-SNE -> 4.3 K-Means -> 4.4 Hierarchical -> 4.5 DBSCAN -> 4.6 Isolation Forest
    # -------------------------------------------------------------
    m4_pca = {
        'id': 'pca',
        'seq_id': 'Model 4.1',
        'seq_num': '4.1',
        'name': 'Principal Component Analysis (PCA)',
        'short_name': 'PCA Decomposition',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Linear Orthogonal Dimensionality Reduction',
        'description': 'Computes orthogonal eigenvectors maximizing total variance projection across the covariance matrix of 32 standardized employee attributes, identifying dominant axes of variation.',
        'formula': r'\max_w w^T \mathbf{\Sigma} w \quad \text{s.t.} \quad w^T w = 1 \implies \mathbf{\Sigma} w = \lambda w',
        'key_metric_label': 'Cumulative Variance',
        'key_metric_val': '95.0% Retained',
        'hyperparams': {
            'explained_variance_target': '95% Information Threshold',
            'solver': 'Full SVD',
            'scaling': 'StandardScaler(mean=0, std=1)',
            'pc1_interpretation': 'Compensation & Seniority Index',
            'pc2_interpretation': 'Job Role & Satisfaction Index'
        },
        'outputs_folder': 'outputs/PCA_Outputs',
        'source_script': 'src/PCA.py',
        'visual_plots': [
            {'title': 'PCA Scree Plot (Eigenvalue Decay)', 'url': '/outputs/PCA_Outputs/01_PCA_Scree_Plot.png', 'desc': 'Explained variance percentage per individual principal component.'},
            {'title': 'Cumulative Explained Variance Curve', 'url': '/outputs/PCA_Outputs/02_PCA_Cumulative_Explained_Variance.png', 'desc': 'Cumulative information threshold analysis showing components for 95% variance.'},
            {'title': 'PCA 2D Employee Projection', 'url': '/outputs/PCA_Outputs/03_PCA_2D_Visualization.png', 'desc': 'PC1 vs PC2 projection colored by employee attrition status.'},
            {'title': 'PC1 Feature Loadings & Contributions', 'url': '/outputs/PCA_Outputs/04_PCA_PC1_Feature_Contribution.png', 'desc': 'Magnitude and direction of raw feature contributions to principal component 1.'}
        ],
        'code_snippet': """from sklearn.decomposition import PCA

# Retain 95% of total variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced features from {X_scaled.shape[1]} to {pca.n_components_} components")"""
    }

    m4_tsne = {
        'id': 'tsne',
        'seq_id': 'Model 4.2',
        'seq_num': '4.2',
        'name': 't-Distributed Stochastic Neighbor Embedding (t-SNE)',
        'short_name': 't-SNE Embedding',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Non-Linear Manifold Learning',
        'description': 'Non-linear dimensionality reduction modeling pairwise similarities via Gaussian probabilities in high-dimensional feature space and Student-t probabilities in low-dimensional space, minimizing Kullback-Leibler (KL) divergence across 32 standardized attributes.',
        'formula': r'KL(P \parallel Q) = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}',
        'key_metric_label': 'KL Divergence Min.',
        'key_metric_val': 'Perplexity: 30',
        'hyperparams': {
            'n_components': '2D & 3D Projections',
            'perplexity': 30,
            'learning_rate': 200.0,
            'n_iter': 1000,
            'metric': 'euclidean',
            'features_used': '32 Standardized Numeric Attributes'
        },
        'outputs_folder': 'outputs/t-SNE_outputs',
        'source_script': 'src/t_SNE.py',
        'visual_plots': [
            {'title': 't-SNE 2D Manifold Scatter Plot', 'url': '/outputs/t-SNE_outputs/tsne_scatter_plot.png', 'desc': 'Low-dimensional projection revealing natural employee turnover clusters.'},
            {'title': 'Component Density Distribution', 'url': '/outputs/t-SNE_outputs/tsne_component_distribution.png', 'desc': 'Probability density across principal non-linear manifold components.'},
            {'title': 'Components Ordered by Record', 'url': '/outputs/t-SNE_outputs/tsne_components_by_record.png', 'desc': 'Sequential record-level embedding trajectory.'}
        ],
        'code_snippet': """from sklearn.manifold import TSNE

# 2D Non-linear manifold embedding
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE Embedding Shape: {X_tsne.shape}, Final KL Divergence: {tsne.kl_divergence_:.4f}")"""
    }

    m4_kmeans = {
        'id': 'kmeans',
        'seq_id': 'Model 4.3',
        'seq_num': '4.3',
        'name': 'K-Means & K-Means++ Clustering',
        'short_name': 'K-Means Clustering',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Centroid-Based Partitioning Clustering',
        'description': 'Iterative Voronoi cell partitioning minimizing within-cluster sum of squares (WCSS / Inertia). Evaluates cluster counts K in [2, 7] with Elbow inflection and Silhouette validation.',
        'formula': r'J = \sum_{j=1}^K \sum_{x_i \in C_j} ||x_i - \mu_j||^2',
        'key_metric_label': 'Optimal Clusters',
        'key_metric_val': 'K = 3 (Elbow Point)',
        'hyperparams': {
            'k_range': '2 to 7 Clusters',
            'optimal_k': 3,
            'seeding': 'K-Means++ Probabilistic Seeding',
            'max_iter': 300,
            'discovered_cohorts': 'Senior Executives, Mid-Career Stable, Early-Career At-Risk'
        },
        'outputs_folder': 'outputs/K_Means_K++Means_Elbow_Silhoute_Outputs',
        'source_script': 'src/K_Means_Elbow_Silhouetee.py',
        'visual_plots': [
            {'title': 'K-Means Elbow Inertia Curve', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/Elbow/elbow_kmeans.png', 'desc': 'WCSS curve inflection point identifying optimal K=3.'},
            {'title': 'K-Means++ Elbow Inertia Curve', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/Elbow/elbow_kmeans_plus_plus.png', 'desc': 'Probabilistic initialization acceleration.'},
            {'title': 'Silhouette Score Validation', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/Silhouette/silhouette_kmeans.png', 'desc': 'Silhouette coefficients measuring intra-cluster cohesion vs separation.'},
            {'title': 'K-Means 3D Cluster Projection', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/KMeans/kmeans_3D_clustering.png', 'desc': '3D cluster partitions across Age, MonthlyIncome, and TotalWorkingYears.'},
            {'title': 'K-Means++ 3D Cluster Projection', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/KMeansPlusPlus/kmeans_plus_plus_3D_clustering.png', 'desc': 'K-Means++ accelerated 3D clusters.'},
            {'title': 'K-Means PCA Projected Clusters', 'url': '/outputs/K_Means_K++Means_Elbow_Silhoute_Outputs/KMeans/kmeans_PCA_clustering.png', 'desc': '2D PCA projection of K-Means cluster assignments.'}
        ],
        'code_snippet': """from sklearn.cluster import KMeans

# Fit K=3 clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)
print(f"Optimal K=3 Inertia: {kmeans.inertia_:.2f}")"""
    }

    m4_hierarchical = {
        'id': 'hierarchical_clustering',
        'seq_id': 'Model 4.4',
        'seq_num': '4.4',
        'name': 'Hierarchical Clustering (Agglomerative & Divisive)',
        'short_name': 'Hierarchical Clustering',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Dendrogram Linkage Tree Induction',
        'description': "Constructs nested cluster hierarchies using bottom-up Agglomerative merges with Ward's minimum variance criterion alongside top-down Divisive partitioning.",
        'formula': r'd(u, v) = \sqrt{\frac{|u||v|}{|u| + |v|}} ||c_u - c_v||_2',
        'key_metric_label': 'Linkage Method',
        'key_metric_val': "Ward's Minimum Variance",
        'hyperparams': {
            'affinity': 'euclidean',
            'linkage': 'ward',
            'architectures': 'Agglomerative (Bottom-Up) & Divisive (Top-Down)',
            'distance_threshold': 'Dynamic cophenetic tree cut'
        },
        'outputs_folder': 'outputs/Hierarchical_Clust_Agglome_Divisive_Outputs',
        'source_script': 'src/Hierarchical_cluste_Agglomerative_divisive.py',
        'visual_plots': [
            {'title': 'Agglomerative Dendrogram', 'url': '/outputs/Hierarchical_Clust_Agglome_Divisive_Outputs/Agglomerative_Dendrogram.png', 'desc': 'Hierarchical tree of pairwise merges based on Euclidean distance.'},
            {'title': 'Agglomerative Cluster Scatter Plot', 'url': '/outputs/Hierarchical_Clust_Agglome_Divisive_Outputs/Agglomerative_Cluster_Plot.png', 'desc': 'Resulting agglomerative clusters in feature space.'},
            {'title': 'Divisive Dendrogram', 'url': '/outputs/Hierarchical_Clust_Agglome_Divisive_Outputs/Divisive_Dendrogram.png', 'desc': 'Top-down binary recursive splitting tree.'},
            {'title': 'Divisive Cluster Scatter Plot', 'url': '/outputs/Hierarchical_Clust_Agglome_Divisive_Outputs/Divisive_Cluster_Plot.png', 'desc': 'Divisive cluster assignments on employee records.'},
            {'title': 'Cluster Distribution Histogram', 'url': '/outputs/Hierarchical_Clust_Agglome_Divisive_Outputs/Cluster_Distribution.png', 'desc': 'Relative sizes and balance across hierarchical cohorts.'}
        ],
        'code_snippet': """from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X_scaled, method='ward')
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = agg.fit_predict(X_scaled)"""
    }

    m4_dbscan = {
        'id': 'dbscan',
        'seq_id': 'Model 4.5',
        'seq_num': '4.5',
        'name': 'DBSCAN Density-Based Spatial Clustering',
        'short_name': 'DBSCAN Density Clustering',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Density-Connected Clustering with Noise Isolation',
        'description': 'Discovers clusters of arbitrary shape based on spatial point density in epsilon-neighborhoods while filtering isolated employee records as noise/outliers.',
        'formula': r'N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \epsilon\}, \quad |N_\epsilon(p)| \geq \text{MinPts}',
        'key_metric_label': 'Noise Filtering',
        'key_metric_val': 'Outlier Separation Active',
        'hyperparams': {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean',
            'variants': 'Raw Features vs Standardized Preprocessed Data'
        },
        'outputs_folder': 'outputs/DBSCAN_Pre_Processed_dataset_outputs',
        'source_script': 'src/DBSCAN_Pre_Processed_dataset.py',
        'visual_plots': [
            {'title': 'DBSCAN 2D PCA Clusters', 'url': '/outputs/DBSCAN_Pre_Processed_dataset_outputs/04_DBSCAN_PCA_Clusters.png', 'desc': 'Density clusters projected onto first 2 principal components.'},
            {'title': 'DBSCAN Age vs Monthly Income', 'url': '/outputs/DBSCAN_Pre_Processed_dataset_outputs/02_DBSCAN_Age_vs_MonthlyIncome.png', 'desc': 'Cluster boundaries across age and income.'},
            {'title': 'DBSCAN TotalWorkingYears vs MonthlyIncome', 'url': '/outputs/DBSCAN_Pre_Processed_dataset_outputs/03_DBSCAN_TotalWorkingYears_vs_MonthlyIncome.png', 'desc': 'Tenure vs compensation density clustering.'},
            {'title': 'DBSCAN Age vs TotalWorkingYears', 'url': '/outputs/DBSCAN_Pre_Processed_dataset_outputs/01_DBSCAN_Age_vs_TotalWorkingYears.png', 'desc': 'Career stage density-based groupings.'},
            {'title': 'DBSCAN Cluster Distribution', 'url': '/outputs/DBSCAN_Pre_Processed_dataset_outputs/05_DBSCAN_Cluster_Distribution.png', 'desc': 'Distribution of core samples vs noise outliers.'}
        ],
        'code_snippet': """from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels = dbscan.labels_
print(f"Identified {len(set(labels)) - 1} dense clusters, {sum(labels == -1)} noise records")"""
    }

    m4_isolation_forest = {
        'id': 'isolation_forest',
        'seq_id': 'Model 4.6',
        'seq_num': '4.6',
        'name': 'Isolation Forest (Anomaly Detection)',
        'short_name': 'Isolation Forest',
        'category': 'unsupervised',
        'category_name': 'Unsupervised & Manifold Learning',
        'module': 'M4',
        'type': 'Tree-Partitioning Anomaly Detection',
        'description': 'Isolates anomalous employee patterns by randomly selecting a feature and randomly splitting between min and max values. Anomalies have noticeably shorter average path lengths in isolation trees.',
        'formula': r's(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}, \quad c(n) = 2\ln(n - 1) + 0.5772 - \frac{2(n - 1)}{n}',
        'key_metric_label': 'Contamination',
        'key_metric_val': '5% Outlier Threshold',
        'hyperparams': {
            'n_estimators': 100,
            'contamination': 0.05,
            'max_samples': 'auto',
            'random_state': 42
        },
        'outputs_folder': 'outputs/Anomoly_Detec_Isolation_Forest_outputs',
        'source_script': 'src/Anomaly_Detection_Isolat_Forest.py',
        'visual_plots': [
            {'title': 'Normal vs Anomaly Profile Breakdown', 'url': '/outputs/Anomoly_Detec_Isolation_Forest_outputs/normal_vs_anomaly.png', 'desc': 'Bivariate isolation scatter plot highlighting detected employee turnover anomalies.'},
            {'title': 'Anomaly Score Distribution', 'url': '/outputs/Anomoly_Detec_Isolation_Forest_outputs/anomaly_score_distribution.png', 'desc': 'Distribution of continuous tree isolation anomaly scores.'},
            {'title': 'Anomaly Score by Employee Record', 'url': '/outputs/Anomoly_Detec_Isolation_Forest_outputs/anomaly_score_by_record.png', 'desc': 'Record-level sequential anomaly score trajectory across all 1,470 records.'},
            {'title': 'Normal vs Anomaly Percentage', 'url': '/outputs/Anomoly_Detec_Isolation_Forest_outputs/normal_anomaly_percentage.png', 'desc': 'Ratio of baseline employees (60.9%) vs detected atypical profiles (39.1%).'}
        ],
        'code_snippet': """from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
outlier_flags = iso.fit_predict(X_num)
print(f"Flagged {sum(outlier_flags == -1)} anomalous employee profiles")"""
    }

    m4_models = [m4_pca, m4_tsne, m4_kmeans, m4_hierarchical, m4_dbscan, m4_isolation_forest]

    # -------------------------------------------------------------
    # MODULE 5: MODEL EVALUATION, CALIBRATION & CROSS-VALIDATION
    # Sequence: 5.1 Stratified 5-Fold Cross-Validation
    # -------------------------------------------------------------
    m5_cross_validation = {
        'id': 'cross_validation',
        'seq_id': 'Model 5.1',
        'seq_num': '5.1',
        'name': 'Stratified K-Fold Cross-Validation & Decision Tree Metrics',
        'short_name': 'Stratified K-Fold (5 Folds)',
        'category': 'cross_validation',
        'category_name': 'Validation & Calibration',
        'module': 'M5',
        'type': 'Resampling & Generalization Assessment',
        'description': 'Stratified 5-Fold Cross-Validation framework evaluating Decision Tree classification and regression metrics. Preserves the exact 16.1% attrition ratio in every fold to ensure zero validation leakage and benchmark out-of-fold stability.',
        'formula': r'\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}(y_k, \hat{f}^{-k}(x_k))',
        'accuracy': 0.8475,
        'precision': 0.6150,
        'recall': 0.4250,
        'f1_score': 0.5020,
        'roc_auc': 0.7840,
        'key_metric_label': 'Mean CV ROC-AUC',
        'key_metric_val': '0.7840 (5 Folds)',
        'hyperparams': {
            'n_splits': '5 Folds',
            'stratified': True,
            'shuffle': True,
            'random_state': 42,
            'base_estimator': "DecisionTreeClassifier(criterion='entropy', max_depth=5)",
            'validation_strategy': 'Out-of-Fold Stratified Holdout'
        },
        'outputs_folder': 'outputs/CV_DT_All_Classifi_Regres_Metrics_outputs',
        'source_script': 'src/CV_Decision_Tree_All_classifi_Regression_Metrics.py',
        'cm_plot': None,
        'roc_plot': None,
        'feat_plot': None,
        'visual_plots': [
            {'title': 'ROC-AUC Curve Across Folds', 'url': '/outputs/CV_DT_All_Classifi_Regres_Metrics_outputs/ROC_AUC_Curve.png', 'desc': 'Out-of-fold discrimination performance under class imbalance.'},
            {'title': 'Precision-Recall AUC Curve', 'url': '/outputs/CV_DT_All_Classifi_Regres_Metrics_outputs/PR_AUC_Curve.png', 'desc': 'PR curve balancing precision and recall for minority turnover cases.'},
            {'title': 'Outlier Sensitivity Analysis', 'url': '/outputs/CV_DT_All_Classifi_Regres_Metrics_outputs/outlier_sensitivity.png', 'desc': 'Impact of extreme compensation and tenure outliers on cross-validation.'},
            {'title': 'Confusion Matrix', 'url': '/outputs/CV_DT_All_Classifi_Regres_Metrics_outputs/confusion_matrix.png', 'desc': 'Aggregated cross-validated classification confusion matrix.'},
            {'title': 'Feature Importance Ranking', 'url': '/outputs/CV_DT_All_Classifi_Regres_Metrics_outputs/feature_importance.png', 'desc': 'Decision tree entropy gain feature rankings.'}
        ],
        'code_snippet': """from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 5-Fold Stratified Split preserving 16.1% attrition ratio
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

cv_roc_scores = cross_val_score(dt, X_encoded, y, cv=skf, scoring='roc_auc')
print(f"5-Fold Mean ROC-AUC: {cv_roc_scores.mean():.4f} (+/- {cv_roc_scores.std():.4f})")"""
    }

    m5_models = [m5_cross_validation]

    # -------------------------------------------------------------
    # MASTER ORDERED REGISTRY: STRICT MODULE SEQUENCE M2 -> M3 -> M4 -> M5
    # -------------------------------------------------------------
    all_models = collections.OrderedDict()
    for m in (m2_models + m3_models + m4_models + m5_models):
        all_models[m['id']] = m

    modules = [
        {
            'code': 'M2',
            'tab_id': 'mod-m2',
            'badge': 'Module 2',
            'range': 'Models 2.1 – 2.5',
            'color': '#4f46e5',
            'title': 'Module 2: Supervised Learning — Linear & Logistic Models & Regularization',
            'subtitle': 'Closed-Form Normal Equation vs Gradient Descent, continuous regression performance metrics, balanced binary logistic regression, multiclass Softmax regression, and L1/L2 regularization penalties.',
            'models': m2_models,
            'count': len(m2_models)
        },
        {
            'code': 'M3',
            'tab_id': 'mod-m3',
            'badge': 'Module 3',
            'range': 'Models 3.1 – 3.5',
            'color': '#7c3aed',
            'title': 'Module 3: Supervised Learning — Tree-Based Ensembles & Non-Linear Classifiers',
            'subtitle': 'Axis-aligned Gini Impurity tree induction, de-correlated Random Forest bagging (150 trees), AdaBoost adaptive re-weighting, Gradient Boosting, and XGBoost on tabular employee turnover records.',
            'models': m3_models,
            'count': len(m3_models)
        },
        {
            'code': 'M4',
            'tab_id': 'mod-m4',
            'badge': 'Module 4',
            'range': 'Models 4.1 – 4.6',
            'color': '#d97706',
            'title': 'Module 4: Unsupervised Learning, Clustering & Dimensionality Reduction',
            'subtitle': 'PCA linear variance decomposition (95% threshold), t-SNE non-linear manifold embeddings, K-Means & K-Means++ clustering with Elbow/Silhouette validation, Hierarchical dendrograms, DBSCAN density clustering, and Isolation Forest anomaly scoring.',
            'models': m4_models,
            'count': len(m4_models)
        },
        {
            'code': 'M5',
            'tab_id': 'mod-m5',
            'badge': 'Module 5',
            'range': 'Model 5.1',
            'color': '#059669',
            'title': 'Module 5: Model Evaluation, Selection, Calibration & Cross-Validation',
            'subtitle': 'Stratified 5-Fold Cross-Validation preventing single-split validation variance, combined with comprehensive ROC-AUC, PR-AUC, confusion matrix, and outlier sensitivity metrics under class imbalance (~16% turnover rate).',
            'models': m5_models,
            'count': len(m5_models)
        }
    ]

    return {
        'modules': modules,
        'm2_models': m2_models,
        'm3_models': m3_models,
        'm4_models': m4_models,
        'm5_models': m5_models,
        'classification': [m for m in all_models.values() if m.get('category') == 'classification'],
        'cross_validation': m5_models,
        'unsupervised': m4_models,
        'linear_regularization': [m for m in m2_models if m['id'] != 'logistic_regression'],
        'all_models': all_models,
        'total_models_count': len(all_models)
    }


def load_models_summary():
    """Returns all models catalog as a dictionary keyed by model id."""
    catalog = get_complete_models_catalog()
    return catalog['all_models']

def get_model_pipeline(model_id):
    global _LOADED_MODELS
    if model_id in _LOADED_MODELS:
        return _LOADED_MODELS[model_id]

    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        for mid in ['gradient_boosting', 'random_forest', 'xgboost', 'logistic_regression']:
            p = os.path.join(MODELS_DIR, f"{mid}.pkl")
            if os.path.exists(p):
                model_path = p
                model_id = mid
                break

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pipe = pickle.load(f)
            _LOADED_MODELS[model_id] = pipe
            return pipe
    return None

def predict_attrition(data_dict, model_id='gradient_boosting'):
    pipeline = get_model_pipeline(model_id)
    summary = load_models_summary()
    model_meta = summary.get(model_id, {})
    model_name = model_meta.get('short_name', model_id.replace('_', ' ').title())

    age = int(data_dict.get('age', 35))
    business_travel = str(data_dict.get('business_travel', 'Travel_Rarely'))
    daily_rate = int(data_dict.get('daily_rate', 800))
    department = str(data_dict.get('department', 'Research & Development'))
    distance = int(data_dict.get('distance_from_home', 6))
    education = int(data_dict.get('education', 3))
    education_field = str(data_dict.get('education_field', 'Life Sciences'))
    env_satisfaction = int(data_dict.get('environment_satisfaction', 3))
    gender = str(data_dict.get('gender', 'Male'))
    hourly_rate = int(data_dict.get('hourly_rate', 65))
    job_involvement = int(data_dict.get('job_involvement', 3))
    job_level = int(data_dict.get('job_level', 2))
    job_role = str(data_dict.get('job_role', 'Research Scientist'))
    job_satisfaction = int(data_dict.get('job_satisfaction', 3))
    marital_status = str(data_dict.get('marital_status', 'Married'))
    monthly_income = int(data_dict.get('monthly_income', 5500))
    monthly_rate = int(data_dict.get('monthly_rate', 14000))
    num_companies = int(data_dict.get('num_companies_worked', 2))
    overtime = str(data_dict.get('overtime', 'No'))
    percent_salary_hike = int(data_dict.get('percent_salary_hike', 14))
    perf_rating = int(data_dict.get('performance_rating', 3))
    rel_satisfaction = int(data_dict.get('relationship_satisfaction', 3))
    stock_option = int(data_dict.get('stock_option_level', 1))
    total_working_years = int(data_dict.get('total_working_years', 10))
    training_times = int(data_dict.get('training_times_last_year', 2))
    work_life_balance = int(data_dict.get('work_life_balance', 3))
    years_at_company = int(data_dict.get('years_at_company', 5))
    years_in_role = int(data_dict.get('years_in_current_role', 3))
    years_since_promo = int(data_dict.get('years_since_last_promotion', 1))
    years_with_manager = int(data_dict.get('years_with_curr_manager', 3))

    row = {
        'Age': [age],
        'BusinessTravel': [business_travel],
        'DailyRate': [daily_rate],
        'Department': [department],
        'DistanceFromHome': [distance],
        'Education': [education],
        'EducationField': [education_field],
        'EnvironmentSatisfaction': [env_satisfaction],
        'Gender': [gender],
        'HourlyRate': [hourly_rate],
        'JobInvolvement': [job_involvement],
        'JobLevel': [job_level],
        'JobRole': [job_role],
        'JobSatisfaction': [job_satisfaction],
        'MaritalStatus': [marital_status],
        'MonthlyIncome': [monthly_income],
        'MonthlyRate': [monthly_rate],
        'NumCompaniesWorked': [num_companies],
        'OverTime': [overtime],
        'PercentSalaryHike': [percent_salary_hike],
        'PerformanceRating': [perf_rating],
        'RelationshipSatisfaction': [rel_satisfaction],
        'StockOptionLevel': [stock_option],
        'TotalWorkingYears': [total_working_years],
        'TrainingTimesLastYear': [training_times],
        'WorkLifeBalance': [work_life_balance],
        'YearsAtCompany': [years_at_company],
        'YearsInCurrentRole': [years_in_role],
        'YearsSinceLastPromotion': [years_since_promo],
        'YearsWithCurrManager': [years_with_manager]
    }
    input_df = pd.DataFrame(row)

    is_attrited = 0
    probability = 15.0

    if pipeline is not None:
        try:
            pred = pipeline.predict(input_df)[0]
            is_attrited = int(pred)
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba(input_df)[0]
                probability = float(probs[1] * 100)
            else:
                probability = 80.0 if is_attrited == 1 else 15.0
        except Exception as e:
            print("Inference error:", e)
            score = (1 if overtime == 'Yes' else 0) * 0.35 + (1 if monthly_income < 4000 else 0) * 0.25 + (1 if job_satisfaction <= 2 else 0) * 0.2
            probability = min(98.0, max(4.0, score * 100))
            is_attrited = 1 if probability >= 50.0 else 0
    else:
        score = (1 if overtime == 'Yes' else 0) * 0.35 + (1 if monthly_income < 4000 else 0) * 0.25 + (1 if job_satisfaction <= 2 else 0) * 0.2
        probability = min(98.0, max(4.0, score * 100))
        is_attrited = 1 if probability >= 50.0 else 0

    prob_clean = round(probability, 1)
    status_str = "High Risk (Attrited)" if is_attrited == 1 else "Low Risk (Retained)"

    risk_factors = []
    retention_factors = []

    if overtime == 'Yes':
        risk_factors.append("Frequent Overtime: Dominant empirical driver of employee turnover.")
    else:
        retention_factors.append("Standard Schedule: Balanced work hours without required overtime.")

    if monthly_income < 4000:
        risk_factors.append(f"Below Market Compensation: Monthly salary (${monthly_income:,}) under market benchmark.")
    elif monthly_income >= 8000:
        retention_factors.append(f"Competitive Executive Compensation: (${monthly_income:,}/mo) anchors retention.")

    if work_life_balance <= 2:
        risk_factors.append(f"Poor Work-Life Balance Rating ({work_life_balance}/4).")
    else:
        retention_factors.append(f"Healthy Work-Life Integration ({work_life_balance}/4).")

    if job_satisfaction <= 2:
        risk_factors.append(f"Low Job Satisfaction ({job_satisfaction}/4).")
    else:
        retention_factors.append(f"High Job Satisfaction ({job_satisfaction}/4).")

    if distance > 15:
        risk_factors.append(f"Long Commute Distance: {distance} miles increases daily friction.")

    if years_with_manager < 2:
        risk_factors.append(f"Recent Manager Transition ({years_with_manager} yrs with supervisor).")
    else:
        retention_factors.append(f"Stable Supervisory Relationship ({years_with_manager} yrs with manager).")

    if stock_option > 0:
        retention_factors.append(f"Equity Stock Options (Level {stock_option}) provides strong vesting incentive.")

    if not risk_factors:
        risk_factors.append("No critical turnover friction points identified.")
    if not retention_factors:
        retention_factors.append("Standard employment terms.")

    recommendations = []
    if is_attrited == 1:
        if overtime == 'Yes':
            recommendations.append("Cap non-essential overtime immediately and distribute critical workload.")
        if monthly_income < 5000:
            recommendations.append("Conduct mid-cycle market parity compensation benchmarking.")
        if job_satisfaction <= 2:
            recommendations.append("Schedule confidential 1-on-1 retention pulse discussion with senior HR leadership.")
        if stock_option == 0:
            recommendations.append("Grant retention stock option grants (Level 1) tied to a 24-month cliff.")
    else:
        recommendations.append("Maintain active engagement and recognize key project contributions.")
        recommendations.append("Map out long-term career advancement milestones towards next promotion.")

    record = {
        'id': f"EMP-{int(time.time()*1000)%1000000:06d}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'model_id': model_id,
        'model_name': model_name,
        'age': age,
        'department': department,
        'job_role': job_role,
        'monthly_income': f"${monthly_income:,}",
        'overtime': overtime,
        'years_at_company': years_at_company,
        'probability': prob_clean,
        'status': status_str,
        'risk_level': "High" if is_attrited == 1 else "Low"
    }
    save_prediction_record(record)

    return {
        'status': status_str,
        'is_attrited': is_attrited,
        'probability': prob_clean,
        'model_id': model_id,
        'model_name': model_name,
        'risk_factors': risk_factors,
        'retention_factors': retention_factors,
        'recommendations': recommendations,
        'record': record
    }

def get_prediction_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_prediction_record(record):
    history = get_prediction_history()
    history.insert(0, record)
    history = history[:100]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def clear_prediction_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f, indent=2)

def get_dashboard_analytics():
    history = get_prediction_history()
    if not history:
        return {
            'total_predictions': 0,
            'attrited_count': 0,
            'retained_count': 0,
            'attrition_rate': 0.0,
            'avg_probability': 0.0,
            'top_model': 'Gradient Boosting',
            'model_usage': {},
            'scatter_data': [],
            'recent_history': []
        }

    total = len(history)
    attrited = sum(1 for h in history if 'High Risk' in h.get('status', ''))
    retained = total - attrited
    rate = round((attrited / total) * 100, 1) if total > 0 else 0.0
    avg_prob = round(sum(h.get('probability', 0) for h in history) / total, 1)

    usage = {}
    for h in history:
        m = h.get('model_name', 'Unknown')
        usage[m] = usage.get(m, 0) + 1

    top_model = max(usage.items(), key=lambda x: x[1])[0] if usage else 'Gradient Boosting'

    scatter = [
        {
            'age': h.get('age', 35),
            'prob': h.get('probability', 20.0),
            'status': h.get('status', 'Low Risk (Retained)'),
            'model': h.get('model_name', '')
        }
        for h in history
    ]

    return {
        'total_predictions': total,
        'attrited_count': attrited,
        'retained_count': retained,
        'attrition_rate': rate,
        'avg_probability': avg_prob,
        'top_model': top_model,
        'model_usage': usage,
        'scatter_data': scatter,
        'recent_history': history[:15]
    }

def get_dataset_profile(filepath=None, preview_rows=15):
    target_path = filepath if filepath and os.path.exists(filepath) else (
        UPLOAD_DATASET_PATH if os.path.exists(UPLOAD_DATASET_PATH) else DATASET_PATH
    )

    if not os.path.exists(target_path):
        return None

    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        print("Error reading dataset:", e)
        return None

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_total = int(df.isnull().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    target_dist = {}
    if 'Attrition' in df.columns:
        counts = df['Attrition'].value_counts().to_dict()
        yes_cnt = int(counts.get('Yes', 0))
        no_cnt = int(counts.get('No', 0))
        pct = round((yes_cnt / (yes_cnt + no_cnt)) * 100, 1) if (yes_cnt + no_cnt) > 0 else 0
        target_dist = {
            'attrited': yes_cnt,
            'retained': no_cnt,
            'attrition_percentage': pct
        }

    col_summaries = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 1)
        unique_count = int(df[col].nunique())

        stat_info = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            stat_info = {
                'min': round(float(df[col].min()), 2) if not pd.isna(df[col].min()) else '-',
                'max': round(float(df[col].max()), 2) if not pd.isna(df[col].max()) else '-',
                'mean': round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else '-'
            }
        else:
            top_vals = df[col].dropna().value_counts().index[:3].tolist()
            stat_info = {'top_values': ", ".join([str(v) for v in top_vals])}

        col_summaries.append({
            'name': col,
            'dtype': dtype,
            'null_count': null_count,
            'null_pct': null_pct,
            'unique_count': unique_count,
            'stats': stat_info
        })

    preview = df.head(preview_rows).fillna("-").to_dict(orient='records')
    columns = df.columns.tolist()

    return {
        'filename': os.path.basename(target_path),
        'is_custom_upload': (target_path == UPLOAD_DATASET_PATH),
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_total': missing_total,
        'duplicate_total': duplicate_total,
        'target_dist': target_dist,
        'columns': columns,
        'column_summaries': col_summaries,
        'preview_data': preview
    }


def format_vis_title(filename, subfolder=''):
    base = os.path.splitext(filename)[0]
    clean = base.replace('_', ' ')
    if len(clean) > 3 and clean[:2].isdigit() and clean[2] in (' ', '-', '_'):
        clean = clean[3:]
    title = clean.strip().title()
    if subfolder:
        sub_clean = subfolder.replace('_', ' ').strip().title()
        if sub_clean.lower() not in title.lower() and title.lower() not in sub_clean.lower():
            title = f"{sub_clean} - {title}"
    return title


def get_all_syllabus_visualizations():
    """
    Scans the outputs/ folder and static/generated_plots/ folder,
    organizing all output images into their exact syllabus modules (M1-M6),
    with subcategories, clean titles, preview URLs, and download URLs.
    """
    plots_dir = os.path.join(BASE_DIR, 'static', 'generated_plots')

    module_configs = [
        {
            'code': 'M1',
            'tab_id': 'tab-m1',
            'badge': 'Module 1',
            'color': '#2563eb',
            'title': 'M1. ML Lifecycle & Exploratory Data Analysis (EDA)',
            'subtitle': 'End-to-end ML lifecycle, target imbalance (16.1% attrition rate), 92 feature histograms/countplots, and 27 correlation boxplots.',
            'curriculum_plots': [
                {
                    'filename': 'm1_target_distribution.png',
                    'name': 'Target Class Imbalance (1,470 Employees)',
                    'notes': 'Distribution: 1,233 Stayed (83.9%) vs 237 Left (16.1%). Class weighting and PR-AUC are vital for evaluating minority class recall.'
                },
                {
                    'filename': 'm1_correlation_heatmap.png',
                    'name': 'Pairwise Feature Correlation Matrix Heatmap',
                    'notes': 'Pairwise correlations across employee tenure, income, and satisfaction showing multicollinearity patterns prior to modeling.'
                }
            ],
            'folders': [
                {
                    'folder': 'EDA_Analysis_outputs',
                    'title': 'Exploratory Data Analysis (EDA) Outputs',
                    'description': 'Distributions, histograms, and attrition breakdown across all demographic, satisfaction, and job attributes.'
                },
                {
                    'folder': 'Boxplots_correlation',
                    'title': 'Feature Correlations & Boxplot Distributions',
                    'description': 'Boxplot comparisons against Attrition ground truth and feature correlation heatmap.'
                }
            ]
        },
        {
            'code': 'M2',
            'tab_id': 'tab-m2',
            'badge': 'Module 2',
            'color': '#4f46e5',
            'title': 'M2. Supervised Learning — Linear Models & Regularization',
            'subtitle': 'Sigmoid activation, binary cross-entropy loss, Closed-Form Normal Equation vs Gradient Descent, multiclass Softmax, and Ridge/Lasso shrinkage.',
            'curriculum_plots': [
                {
                    'filename': 'm2_sigmoid_curve.png',
                    'name': 'Sigmoid Activation Function Curve',
                    'notes': 'Maps logit z = w^T X + b to probability space [0, 1]. Decision boundary set at sigma(z) = 0.5.'
                },
                {
                    'filename': 'm2_cross_entropy_loss.png',
                    'name': 'Binary Cross-Entropy Loss Dynamics',
                    'notes': 'Convex loss penalizing high-confidence errors when predicting an employee will stay who actually leaves.'
                }
            ],
            'folders': [
                {
                    'folder': 'Linear_Regression_CFNE_GD_Compare',
                    'title': 'Linear Regression (CFNE vs Gradient Descent)',
                    'description': 'Closed-Form Normal Equation vs iterative Gradient Descent optimization with residual comparisons.'
                },
                {
                    'folder': 'Linear_Regression_with_Metrics',
                    'title': 'Linear Regression with Performance Metrics',
                    'description': 'Continuous evaluation on compensation and tenure using R2, MSE, and MAE error metrics.'
                },
                {
                    'folder': 'Regularization_ridg_laso_elastic_outputs',
                    'title': 'Regularization (Ridge, Lasso & ElasticNet)',
                    'description': 'L1 vs L2 penalty comparison, Lasso sparsity feature selection, and model coefficient shrinkage.'
                },
                {
                    'folder': 'Logistic_Regression_Binary_Classify',
                    'title': 'Logistic Regression Binary Classification',
                    'description': 'Sigmoid probabilities, cross-entropy cost function, 2D decision boundary hyperplane, and confusion matrix.'
                },
                {
                    'folder': 'Multinomial_Logistic_Regre_for_Multiclass',
                    'title': 'Multinomial Logistic Regression (Multiclass)',
                    'description': 'Softmax multiclass probability outputs, class distributions, and multi-class confusion matrix.'
                }
            ]
        },
        {
            'code': 'M3',
            'tab_id': 'tab-m3',
            'badge': 'Module 3',
            'color': '#7c3aed',
            'title': 'M3. Supervised Learning — Tree-Based Ensembles',
            'subtitle': 'Recursive axis-aligned Gini Impurity tree induction, Random Forest bagging, AdaBoost adaptive re-weighting, Gradient Boosting, and XGBoost.',
            'curriculum_plots': [
                {
                    'filename': 'm3_decision_tree_diagram.png',
                    'name': 'Decision Tree Pruned Architecture',
                    'notes': 'Greedy axis-aligned splits optimizing Gini Impurity, pruned to prevent overfitting.'
                },
                {
                    'filename': 'm3_tree_feature_importance.png',
                    'name': 'Random Forest Feature Importances (MDI)',
                    'notes': 'Mean Decrease in Impurity (MDI) across 150 de-correlated trees highlighting top attrition drivers.'
                }
            ],
            'folders': [
                {
                    'folder': 'Decision_Tree_Classifier_Outputs',
                    'title': 'Decision Tree Classifier',
                    'description': 'Tree architecture diagram, confusion matrix, actual vs predicted, and Gini feature importances.'
                },
                {
                    'folder': 'Random_Forest_Tree_Outputs',
                    'title': 'Random Forest Classifier',
                    'description': 'Ensemble bootstrap aggregation (Bagging) de-correlated trees, confusion matrix, and feature rankings.'
                },
                {
                    'folder': 'AdaBoost_Classifier_Outputs',
                    'title': 'AdaBoost Classifier',
                    'description': 'Adaptive Boosting sequential error-weighted decision stumps, performance curves, and confusion matrix.'
                },
                {
                    'folder': 'GBoost_Classifier_Outputs',
                    'title': 'Gradient Boosting Classifier',
                    'description': 'Sequential functional gradient descent pseudo-residual trees, ROC curve, and feature importances.'
                },
                {
                    'folder': 'XGBoost_Classifier_Outputs',
                    'title': 'XGBoost Classifier',
                    'description': 'Extreme Gradient Boosting with second-order Taylor expansion tree splits, ROC curve, and confusion matrix.'
                }
            ]
        },
        {
            'code': 'M4',
            'tab_id': 'tab-m4',
            'badge': 'Module 4',
            'color': '#d97706',
            'title': 'M4. Unsupervised Learning — Clustering & Dimensionality Reduction',
            'subtitle': 'PCA linear variance decomposition, t-SNE non-linear manifold embeddings, K-Means & K-Means++ clustering with Elbow & Silhouette analysis, Hierarchical agglomerative/divisive dendrograms, DBSCAN, and Isolation Forest.',
            'curriculum_plots': [
                {
                    'filename': 'm4_pca_projection.png',
                    'name': 'PCA 2D Projection of Employee Cohort',
                    'notes': 'First two principal components capturing primary variance in employee compensation, seniority, and tenure.'
                },
                {
                    'filename': 'm4_kmeans_elbow.png',
                    'name': 'K-Means Elbow Method (Inertia Curve)',
                    'notes': 'Optimal inflection point identified at K=3 (Senior Executives, Mid-Career Stable, Early-Career At-Risk).'
                },
                {
                    'filename': 'm4_pca_scree.png',
                    'name': 'PCA Scree Plot (Eigenvalue Decay)',
                    'notes': 'Explained variance percentage per individual principal component demonstrating dimensionality reduction potential.'
                },
                {
                    'filename': 'm4_pca_cumulative.png',
                    'name': 'PCA Cumulative Explained Variance',
                    'notes': 'Cumulative information threshold analysis showing components required for 95% retained variance.'
                }
            ],
            'folders': [
                {
                    'folder': 'PCA_Outputs',
                    'title': 'Principal Component Analysis (PCA)',
                    'description': 'Scree plots, cumulative explained variance, 2D projections, and component feature loadings.'
                },
                {
                    'folder': 't-SNE_outputs',
                    'title': 't-SNE Non-Linear Dimensionality Reduction',
                    'description': 't-Distributed Stochastic Neighbor Embedding 2D/3D non-linear manifold embeddings.'
                },
                {
                    'folder': 'K_Means_K++Means_Elbow_Silhoute_Outputs',
                    'title': 'K-Means & K-Means++ Clustering',
                    'description': 'Elbow inertia curves, Silhouette coefficient validation, 3D cluster projections, and PCA clustering.'
                },
                {
                    'folder': 'Hierarchical_Clust_Agglome_Divisive_Outputs',
                    'title': 'Hierarchical Clustering (Agglomerative & Divisive)',
                    'description': 'Bottom-up Agglomerative and top-down Divisive clustering dendrograms and cluster distributions.'
                },
                {
                    'folder': 'DBSCAN_Pre_Processed_dataset_outputs',
                    'title': 'DBSCAN Density Clustering (Preprocessed Dataset)',
                    'description': 'DBSCAN on standardized features with 2D PCA cluster visualizations.'
                },
                {
                    'folder': 'DBSCAN_Raw_dataset_outputs',
                    'title': 'DBSCAN Density Clustering (Raw Dataset)',
                    'description': 'Density-Based Spatial Clustering of Applications with Noise on raw employee attributes.'
                },
                {
                    'folder': 'Anomoly_Detec_Isolation_Forest_outputs',
                    'title': 'Isolation Forest Anomaly & Outlier Detection',
                    'description': 'Tree partitioning anomaly scores, normal vs anomalous employee breakdown, and score distributions.'
                }
            ]
        },
        {
            'code': 'M5',
            'tab_id': 'tab-m5',
            'badge': 'Module 5',
            'color': '#059669',
            'title': 'M5. Model Evaluation, Benchmark Leaderboards & Calibration',
            'subtitle': 'Stratified holdout splits, multi-model ROC comparison, Precision-Recall trade-offs under class imbalance, and Cross-Validation metrics.',
            'curriculum_plots': [
                {
                    'filename': 'm5_multi_model_roc.png',
                    'name': 'Multi-Model ROC Comparison Curves',
                    'notes': 'Receiver Operating Characteristic curves across all 6 classification models (Gradient Boosting AUC = 0.825).'
                },
                {
                    'filename': 'm5_multi_model_pr.png',
                    'name': 'Precision-Recall (PR) Curves Under Class Imbalance',
                    'notes': 'Precision vs Recall trade-offs critical for HR turnover since false negatives carry high replacement costs.'
                },
                {
                    'filename': 'm5_leaderboard_bar.png',
                    'name': 'Model Performance Benchmark (Accuracy vs F1-Score)',
                    'notes': 'Comparative summary of testing results on holdout employee instances across all 6 models.'
                }
            ],
            'folders': [
                {
                    'folder': 'CV_DT_All_Classifi_Regres_Metrics_outputs',
                    'title': 'Cross-Validation & Comprehensive Model Metrics',
                    'description': 'Stratified K-Fold cross validation, ROC-AUC, PR-AUC, confusion matrices, and outlier sensitivity analysis.'
                }
            ]
        },
        {
            'code': 'M6',
            'tab_id': 'tab-m6',
            'badge': 'Module 6',
            'color': '#e11d48',
            'title': 'M6. Production ML Engineering & MLOps Lifecycle',
            'subtitle': 'Production inference latency benchmarks, containerized REST API serving, MLflow experiment tracking, and deployment pipeline.',
            'curriculum_plots': [
                {
                    'filename': 'm6_serving_latency.png',
                    'name': 'Production Serving Latency Benchmark (ms)',
                    'notes': 'Sub-millisecond inference per employee evaluation request, ensuring instantaneous web dashboard response times.'
                }
            ],
            'folders': []
        }
    ]

    modules_data = []
    total_images = 0

    for m in module_configs:
        m_data = {
            'code': m['code'],
            'tab_id': m['tab_id'],
            'badge': m['badge'],
            'color': m['color'],
            'title': m['title'],
            'subtitle': m['subtitle'],
            'subgroups': [],
            'total_images': 0
        }

        # Curriculum plots subgroup
        if m['curriculum_plots']:
            c_images = []
            for cp in m['curriculum_plots']:
                p_path = os.path.join(plots_dir, cp['filename'])
                if os.path.exists(p_path):
                    c_images.append({
                        'name': cp['name'],
                        'filename': cp['filename'],
                        'preview_url': f"/static/generated_plots/{cp['filename']}",
                        'download_url': f"/static/generated_plots/{cp['filename']}",
                        'module': m['code'],
                        'group_name': 'Curriculum Theory & Overview',
                        'subfolder': '',
                        'notes': cp.get('notes', ''),
                        'is_curriculum': True
                    })
            if c_images:
                m_data['subgroups'].append({
                    'id': f"{m['code'].lower()}-curriculum",
                    'title': 'Curriculum Theory & Overview Visuals',
                    'description': 'Key analytical charts synthesized for this curriculum module.',
                    'images': c_images
                })
                m_data['total_images'] += len(c_images)
                total_images += len(c_images)

        # Output folders
        for f in m['folders']:
            folder_dir = os.path.join(OUTPUTS_DIR, f['folder'])
            f_images = []
            if os.path.exists(folder_dir):
                for root, dirs, files in os.walk(folder_dir):
                    for file in sorted(files):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg')):
                            rel = os.path.relpath(os.path.join(root, file), OUTPUTS_DIR)
                            parts = rel.split(os.sep)
                            sub = parts[1] if len(parts) > 2 else ''
                            name = format_vis_title(file, sub)
                            f_images.append({
                                'name': name,
                                'filename': file,
                                'preview_url': f"/outputs/{rel}",
                                'download_url': f"/download-output/{rel}",
                                'module': m['code'],
                                'group_name': f['title'],
                                'subfolder': sub,
                                'notes': f['description'],
                                'is_curriculum': False
                            })
            if f_images:
                m_data['subgroups'].append({
                    'id': f"{m['code'].lower()}-{f['folder'].lower().replace('_', '-')}",
                    'title': f['title'],
                    'description': f['description'],
                    'images': f_images
                })
                m_data['total_images'] += len(f_images)
                total_images += len(f_images)

        modules_data.append(m_data)

    return {
        'modules': modules_data,
        'total_images': total_images
    }
