from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin

"""
Canonical Correlation Analysis: CCA searches for a shared representation of the EEG and video data.
It does not directly learn a multiclass decision boundary; it learns a relationship between two
different modalities.
"""

class CCA_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 2, pca_components: int = 30, max_iter: int = 1000):
        self.n_components = n_components
        self.pca_components = pca_components
        self.max_iter = max_iter
        self.eeg_scaler = None
        self.video_scaler = None
        self.eeg_pca = None
        self.video_pca = None
        self.cca = None

    def fit(self, X_train, Z_train):
        # 1. Scale features independently
        self.eeg_scaler = StandardScaler().fit(X_train)
        self.video_scaler = StandardScaler().fit(Z_train)
        
        X_scaled = self.eeg_scaler.transform(X_train)
        Z_scaled = self.video_scaler.transform(Z_train)
        
        # 2. Apply PCA to mitigate high dimensionality & overfitting
        n_eeg_pc = min(self.pca_components, X_scaled.shape[0], X_scaled.shape[1])
        n_video_pc = min(self.pca_components, Z_scaled.shape[0], Z_scaled.shape[1])
        
        self.eeg_pca = PCA(n_components=n_eeg_pc).fit(X_scaled)
        self.video_pca = PCA(n_components=n_video_pc).fit(Z_scaled)
        
        X_pca = self.eeg_pca.transform(X_scaled)
        Z_pca = self.video_pca.transform(Z_scaled)
        
        # 3. Fit CCA on PCA-reduced spaces
        actual_components = min(self.n_components, n_eeg_pc, n_video_pc)
        self.cca = CCA(n_components=actual_components, scale=False, max_iter=self.max_iter)
        self.cca.fit(X_pca, Z_pca)
        return self

    def predict(self, eeg_features, candidate_video_features):
        n_candidates = len(candidate_video_features)
        repeated_eeg = np.repeat(eeg_features, repeats=n_candidates, axis=0)
        
        # Transform through the exact same pipeline steps
        eeg_scaled = self.eeg_scaler.transform(repeated_eeg)
        video_scaled = self.video_scaler.transform(candidate_video_features)
        
        eeg_pca = self.eeg_pca.transform(eeg_scaled)
        video_pca = self.video_pca.transform(video_scaled)
        
        eeg_proj, video_proj = self.cca.transform(eeg_pca, video_pca)

        eeg_proj = normalize(eeg_proj, axis=1)
        video_proj = normalize(video_proj, axis=1)
        scores = np.sum(eeg_proj * video_proj, axis=1)

        predicted_index = int(np.argmax(scores))
        return predicted_index, scores

def _fit(X, Y, video_ids, video_features, params):
    row_of_id = {video_id: row for row, video_id in enumerate(video_ids)}
    Z = video_features[[row_of_id[video_id] for video_id in Y]]

    model = CCA_Classifier(**params).fit(X, Z)
    return model 

def _accuracy(model, X, Y, video_ids, video_features):
    correct = 0
    for eeg_features, video_id in zip(X, Y):
        predicted_index, _ = model.predict(eeg_features[None, :], video_features)
        correct += video_ids[predicted_index] == video_id
    print(f"Accuracy: {correct / len(Y)}")
    return correct / len(Y)

def evaluate_cca(X_train, Y_train, train_video_ids, train_video_features,
                  X_test, Y_test, test_video_ids, test_video_features, 
                  random_seed: int = 42):
    """
    Data is split into train/validation/test, hyper parameter tuning of CCA,
    best parameters are returned.

    X_train: (n_trials, n_eeg_features)
    Y_train: (n_trials,) id of the video watched in each trial
    train_video_ids: (n_videos,) ids of the candidate videos, same values as Y
    train_video_features: (n_videos, n_video_features), row i belongs to video_ids[i]

    X_test: (n_trials, n_eeg_features)
    Y_test: (n_trials,) id of the video watched in each trial
    test_video_ids: (n_videos,) ids of the candidate videos, same values as Y
    test_video_features: (n_videos, n_video_features), row i belongs to video_ids[i]
    """
    train_video_ids = np.asarray(train_video_ids)
    test_video_ids = np.asanyarray(test_video_ids)

    train_video_features = np.asarray(train_video_features)
    test_video_features = np.asarray(test_video_features)

    param_grid = {
        #"n_components": [2, 4, 8, 16, 32],
        "n_components": [2,4,8],
    }

    # Grid_search over possible parameter variations via 5-fold cross validation.
    # GridSearchCV cannot be used, because CCA_Classifier.predict needs the candidates.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    best_params, best_score = None, -np.inf
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, Y_train):
            model = _fit(X_train[train_idx], Y_train[train_idx], train_video_ids, train_video_features, params)
            fold_scores.append(_accuracy(model, X_train[val_idx], Y_train[val_idx], train_video_ids, train_video_features))

        score = float(np.mean(fold_scores))
        print(f"{params}: CV accuracy {score:.4f}")
        if score > best_score:
            best_params, best_score = params, score

    print("Best parameters:", best_params)
    print("Best CV accuracy:", best_score)

    model = _fit(X_train, Y_train, train_video_ids, train_video_features, best_params)
    test_accuracy = _accuracy(model, X_test, Y_test, test_video_ids, test_video_features)

    print("Test accuracy:", test_accuracy)
    print("Chance level:", 1 / len(test_video_ids))

    return model, best_params

def save_submission_ranked_list(model, X_test, test_video_ids, test_video_features, output_path):

    # output_path = "../predictions/sub-05_ranked_candidate_list.npy"
    # output_path = "../predictions/sub-06_ranked_candidate_list.npy"

    test_video_ids = np.asarray(test_video_ids, dtype=np.int_)
    test_video_features = np.asarray(test_video_features)
    
    ranked_candidate_matrix = []
    
    for eeg_features in X_test:
        _, scores = model.predict(eeg_features[None, :], test_video_features)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Map indices to actual video IDs (ensuring they are integer IDs like 1001-1102)
        sorted_video_ids = test_video_ids[sorted_indices]
        
        ranked_candidate_matrix.append(sorted_video_ids)
        
    ranked_candidate_array = np.array(ranked_candidate_matrix, dtype=np.int_)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ranked_candidate_array)
    print(f"Successfully saved ranked candidate list of shape {ranked_candidate_array.shape} to {output_path}")
    
    return ranked_candidate_array