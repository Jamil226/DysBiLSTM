"""scikit-learn baselines (Table V).

Every model is a Pipeline whose scaler (and PCA, where used) is fitted on
the training partition only.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .. import config


def _pipe(clf, pca: bool = False) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=config.SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


# name -> (factory, feature type)
CLASSICAL_MODELS = {
    "dt": (lambda: _pipe(DecisionTreeClassifier(
        max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=config.SEED)), "mfcc"),
    "rf": (lambda: _pipe(RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=2,
        random_state=config.SEED, n_jobs=-1)), "mfcc"),
    "svm": (lambda: _pipe(SVC(
        kernel="rbf", C=10, gamma="scale", probability=True, random_state=config.SEED)), "mfcc"),
    "nb_flat": (lambda: _pipe(GaussianNB()), "mfcc"),
    "nb_stat": (lambda: _pipe(GaussianNB(var_smoothing=1e-8), pca=True), "stats"),
    "lr": (lambda: _pipe(LogisticRegression(
        C=2.0, solver="lbfgs", max_iter=1000), pca=True), "stats"),
}
