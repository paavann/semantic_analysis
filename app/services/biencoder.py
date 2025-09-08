import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


    
class BiEncoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        logger.info(f"bi-encoder model {model_name} loaded successfully")

    def score(self, chunks: List[str], topic: str) -> np.ndarray:
        try:
            chunk_embs = self.model.encode(chunks, normalize_embeddings=True)
            topic_emb = self.model.encode([topic], normalize_embeddings=True)
            return cosine_similarity(topic_emb, chunk_embs)[0]
        except Exception as e:
            logger.error(f"bi-encoder scoring failed: {e}")
            return np.zeros(len(chunks))