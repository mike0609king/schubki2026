from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, max_depth: int, random_seed: int = 42):
        self.model = RandomForestClassifier(max_depth=max_depth, random_state=random_seed)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

# TODO
def evaluate_random_forest(self, X, Y):
    """
    Data is split into train/validation/test, hyper parameter tuning of random forrest,
    best parameters are returned.
    """
    # hyper parameters: 
    # 1. max_depth
    raise NotImplementedError
