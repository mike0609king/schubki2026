from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 1, n_estimators: int = 100, random_seed: int = 42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_seed = random_seed

    def fit(self, X, Y):
        self.model = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_seed)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def evaluate_random_forest(X, Y, random_seed: int = 42):
    """
    Data is split into train/validation/test, hyper parameter tuning of random forest,
    best parameters are returned.
    """

    pipeline = Pipeline([
        ("rforest", RandomForest())
        ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_seed, stratify=Y)

    param_grid = {
        "rforest__max_depth": [1, 5, 10, 20, 30],
        "rforest__n_estimators": [100, 200, 300]
    }

    # Grid_search over possible parameter variations via 5-fold cross validation
    grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=10, scoring="accuracy", cv=5, refit=True, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    Y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print("Test accuracy:", test_accuracy)

    return grid_search.best_params_

