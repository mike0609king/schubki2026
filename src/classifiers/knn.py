from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, Y):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def evaluate_knn(X, Y, random_seed: int = 42):
    """
    Data is split into train/validation/test, hyper parameter tuning of KNN,
    best parameters are returned.
    """

    pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("knn", KNN())
        ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_seed, stratify=Y)

    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 41, 51],
        "knn__weights": ["uniform", "distance"],
    }

    # Grid_search over possible parameter variations via 5-fold cross validation
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    Y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    print("Test accuracy:", test_accuracy)

    return grid_search.best_params_
