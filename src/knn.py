from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, n_neighbors: int):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

# TODO
def evaluate_knn(self, X, Y):
    """
    Data is split into train/validation/test, hyper parameter tuning of KNN,
    best parameters are returned.
    """
    # hyper parameters: 
    # 1. n_neighbors
    # 2. StandardScalar()
    raise NotImplementedError
