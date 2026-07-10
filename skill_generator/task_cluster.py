from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from models import Trajectory
from typing import List

class Clusterer:
    def __init__(self, n_clusters=20, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit_predict(self, trajs: List[Trajectory]):
        # 将 domain 作为前缀，强化领域一致性（这里 domain 已确保相同），使用 KMeans 方法进行语义聚类
        texts = [f"[{t.domain}] {t.instruction}" for t in trajs]
        embeddings = self.encoder.encode(texts)
        
        labels = self.kmeans.fit_predict(embeddings)
        clusters = {i: [] for i in range(self.kmeans.n_clusters)}
        
        for idx, label in enumerate(labels):
            clusters[label].append(trajs[idx])
        
        return clusters
