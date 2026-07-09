from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

class TaskClusterer:
    def __init__(self, model_name='all-MiniLM-L6-v2', n_clusters=20):
        self.encoder = SentenceTransformer(model_name)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit(self, trajectories: list[dict]):
        # trajectories: list of dict with 'instruction', 'app', 'type'
        texts = [f"{t['instruction']}" for t in trajectories]
        embeddings = self.encoder.encode(texts)
        labels = self.kmeans.fit_predict(embeddings)
        
        clusters = {i: [] for i in range(self.kmeans.n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(trajectories[idx])
        
        return clusters