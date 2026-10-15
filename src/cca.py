import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

"""
Canonical Correlation Analysis: CCA searches for a shared representation of the EEG and video data. 
It does not directly learn a multiclass decision boundary; it learns a relationship between two 
different modalities.
"""

class CCA_Classifier():
    def __init__(self, n_components: int = 2, max_iter: int = 1000):
        self.n_components = n_components
        self.max_iter = max_iter
        self.cca = None

    def fit(self, X_train, Z_train):
        self.cca = CCA(n_components=self.n_components, scale=True, max_iter=self.max_iter)
        self.cca.fit(X_train, Z_train)
        return self

    def predict(self, eeg_features, candidate_video_featues):
        """
        eeg_features: (1, n_eeg_features)
        candidate_video_features: (n_candidates, n_video_features)

        Returns:
            predicted_index: index of the best candidate
            scores: similarity scores for all candidates
        """
        if self.cca is None:
            raise RuntimeError("Call fit() before predict() in CCA_Classifier.")

        n_candidates = len(candidate_video_featues)

        repeated_eeg = np.repeat(eeg_features, repeats=n_candidates, axis=0)
        eeg_proj, video_proj = self.cca.transform(repeated_eeg, candidate_video_featues)

        # Cosine Sim
        eeg_proj = normalize(eeg_proj, axis=1)
        video_proj = normalize(video_proj, axis=1)
        scores = np.sum(eeg_proj * video_proj, axis=1)

        predicted_index = int(np.argmax(scores))

        return predicted_index, scores