"""
Loan Default Prediction — Portfolio Code Extract
Author: Joseph Kayefor Ebigwai

This file is a cleaned portfolio extract adapted from the author's
Loan Default Prediction project notebook. It demonstrates the
leakage-safe preprocessing, random oversampling, Gradient Boosting,
and hyperparameter-tuning design used in the project.

The original course dataset is intentionally not included.
"""

from scipy.stats import randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def build_preprocessor(numerical_features, categorical_features):
    """Create a leakage-safe preprocessing template."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # sparse_output is used by newer scikit-learn versions.
    try:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse=False,
                    ),
                ),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_oversampled_gradient_boosting_pipeline(preprocessor):
    """Combine preprocessing, oversampling and Gradient Boosting."""

    return ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "sampler",
                RandomOverSampler(random_state=42),
            ),
            (
                "model",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    )


def build_random_search(model_pipeline, cross_validation):
    """Create the randomized hyperparameter search used for model tuning."""

    parameter_distributions = {
        "model__n_estimators": randint(100, 501),
        "model__learning_rate": uniform(0.02, 0.18),
        "model__max_depth": randint(1, 5),
        "model__min_samples_split": randint(2, 16),
        "model__min_samples_leaf": randint(1, 8),
        "model__subsample": uniform(0.65, 0.35),
        "model__max_features": [None, "sqrt", "log2", 0.5, 0.75],
    }

    return RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=parameter_distributions,
        n_iter=30,
        scoring="average_precision",
        cv=cross_validation,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )


# Example workflow:
#
# preprocessor = build_preprocessor(
#     numerical_features=numerical_features,
#     categorical_features=categorical_features,
# )
#
# model_pipeline = build_oversampled_gradient_boosting_pipeline(
#     preprocessor
# )
#
# random_search = build_random_search(
#     model_pipeline,
#     cross_validation=candidate_cv,
# )
#
# random_search.fit(X_train_fe, y_train)
#
# print("Best CV PR-AUC:", round(random_search.best_score_, 4))
# print("Best parameters:", random_search.best_params_)
