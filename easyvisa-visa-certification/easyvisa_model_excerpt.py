"""
EasyVisa – Visa Certification Prediction
Portfolio Code Extract

Author: Joseph Kayefor Ebigwai

This cleaned extract is based on the modelling workflow developed in the
EasyVisa project. It demonstrates class-imbalance treatment with SMOTE,
ensemble model comparison using F1-score, and Random Forest hyperparameter
tuning with GridSearchCV.

The original course dataset is intentionally not included.
"""

import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def oversample_training_data(X_train, y_train):
    """Balance the training data with SMOTE."""
    smote = SMOTE(
        sampling_strategy=1,
        k_neighbors=5,
        random_state=1,
    )
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
    return X_train_over, y_train_over


def compare_ensemble_models(X_train_over, y_train_over, X_val, y_val):
    """Compare ensemble classifiers using validation F1-score."""
    models = [
        ("Bagging", BaggingClassifier(random_state=1)),
        ("Random Forest", RandomForestClassifier(random_state=1)),
        ("AdaBoost", AdaBoostClassifier(random_state=1)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=1)),
    ]

    if XGBClassifier is not None:
        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    random_state=1,
                    eval_metric="logloss",
                ),
            )
        )

    results = {}

    for name, model in models:
        model.fit(X_train_over, y_train_over)
        validation_predictions = model.predict(X_val)
        results[name] = f1_score(y_val, validation_predictions)

    return results


def tune_random_forest(X_train_over, y_train_over):
    """
    Tune Random Forest on oversampled training data using 5-fold
    cross-validation and F1-score.
    """
    random_forest = RandomForestClassifier(
        random_state=1,
        oob_score=True,
        bootstrap=True,
    )

    parameters = {
        "max_depth": list(np.arange(5, 15, 5)),
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [3, 5, 7],
        "n_estimators": list(np.arange(10, 40, 10)),
    }

    f1_scorer = make_scorer(f1_score)

    search = GridSearchCV(
        estimator=random_forest,
        param_grid=parameters,
        scoring=f1_scorer,
        cv=5,
        n_jobs=-1,
    )

    search.fit(X_train_over, y_train_over)

    return search.best_estimator_, search.best_params_, search.best_score_


# Example workflow:
#
# X_train_over, y_train_over = oversample_training_data(X_train, y_train)
#
# model_scores = compare_ensemble_models(
#     X_train_over,
#     y_train_over,
#     X_val,
#     y_val,
# )
#
# best_rf, best_params, best_cv_f1 = tune_random_forest(
#     X_train_over,
#     y_train_over,
# )
#
# test_predictions = best_rf.predict(X_test)
# test_f1 = f1_score(y_test, test_predictions)
