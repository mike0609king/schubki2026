from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

USE_CUML = True
if USE_CUML:
    from cuml.svm import SVC as SVM
else:
    from sklearn.svm import SVC as SVM

class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C: float = 1, kernel: str = "linear", random_seed: int = 42):
        self.C = C
        self.kernel = kernel
        self.max_iter = -1
        self.random_seed = random_seed

    def fit(self, X, Y):

        self.model = SVM(C=self.C, kernel=self.kernel, max_iter=self.max_iter, random_state=self.random_seed)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def evaluate_svc(X, Y, random_seed: int = 42):
    """
    Data is split into train/validation/test, hyper parameter tuning of SVC,
    best parameters are returned.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC())
        ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_seed, stratify=Y)

    param_grid = {
        "svc__C": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        "svc__kernel": ["linear", "rbf"],
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
