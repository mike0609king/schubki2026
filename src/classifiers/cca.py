from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, KFold
from sklearn.base import BaseEstimator, ClassifierMixin

"""
Canonical Correlation Analysis: CCA searches for a shared representation of the EEG and video data.
It does not directly learn a multiclass decision boundary; it learns a relationship between two
different modalities.
"""
class CCA_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 8, pca_components: int = 30, max_iter: int = 1000):
        self.n_components = n_components
        self.pca_components = pca_components
        self.max_iter = max_iter
        self.cca = None

    def fit(self, X_train, Z_train):
        actual_components = min(self.n_components, X_train.shape[1], Z_train.shape[1])
        self.cca = CCA(n_components=actual_components, scale=False, max_iter=self.max_iter)
        self.cca.fit(X_train, Z_train)
        return self

    def predict(self, eeg_features, candidate_video_features):
        scores_matrix = self.get_score_matrix(eeg_features, candidate_video_features)
        predicted_index = int(np.argmax(scores_matrix[0]))
        return predicted_index, scores_matrix[0]

    def get_score_matrix(self, X_test, candidate_video_features):
        eeg_proj, video_proj = self.cca.transform(X_test, candidate_video_features)
        eeg_proj = normalize(eeg_proj, axis=1)
        video_proj = normalize(video_proj, axis=1)
        score_matrix = np.dot(eeg_proj, video_proj.T)
        return score_matrix
