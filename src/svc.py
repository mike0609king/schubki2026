from sklearn.svm import LinearSVC

class SVC:
    def __init__(self, C: float, max_iter: int, random_seed: int = 42):
        self.model = LinearSVC(C=C, max_iter=max_iter, random_state=random_seed)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

# TODO
def evaluate_svc(self, X, Y):
    """
    Data is split into train/validation/test, hyper parameter tuning of SVC,
    best parameters are returned.
    """
    # hyper parameters: 
    # 1. C
    # 2. max_iter
    # 3. StandardScalar()
    # 4. class_weight, "balanced" for imbalanced dataset
    raise NotImplementedError
