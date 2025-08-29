from typing import List, Tuple
import logging

import numpy as np

from utils.download_punkt import download_punkt
download_punkt()
from nltk.tokenize.punkt import PunktSentenceTokenizer

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

from models.relevance_model import Relevance


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicRelevanceScorer:
    def __init__(self,
                 bi_encoder_model: str="paraphrase-mpnet-base-v2",
                 max_chunk_chars: int=250,):
        self.max_chunk_chars = max_chunk_chars

        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        logger.info(f"bi-encoder model {bi_encoder_model} loaded successfully")
    

    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        
        tokenizer = PunktSentenceTokenizer()
        try:
            sentences = tokenizer.tokenize(text)
        except Exception as e:
            logger.warning(f"punkt tokenizer failed: {e}. using fallback split")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks, current_chunk = [], ""
        for sentence in sentences:
            potential_chunk = (current_chunk + " " + sentence).strip()

            if len(potential_chunk) <= self.max_chunk_chars:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
  
                if len(sentence) <= self.max_chunk_chars:
                    current_chunk = sentence
                else:
                    words = sentence.split()
                    temp_chunk = ""

                    for word in words:
                        if len(temp_chunk + " " + word) <= self.max_chunk_chars:
                            temp_chunk = (temp_chunk + " " + word).strip()
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word

                    current_chunk = temp_chunk

        if current_chunk:
            chunks.append(current_chunk)

        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 10]
        logger.info(f"text split into {len(chunks)} chunks")
        return chunks
    

    def _bi_encoder_scoring(self, chunks: List[str], topic: str) -> np.ndarray:
        try:
            chunk_embs = self.bi_encoder.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
            topic_emb = self.bi_encoder.encode([topic], convert_to_tensor=False, normalize_embeddings=True)

            similarities = cosine_similarity(topic_emb, chunk_embs)[0]
            return similarities
        except Exception as e:
            logger.error(f"bi-encoder scoring failed: {e}")
            return np.zeros(len(chunks))
        
    
    def _calculate_relevance_metrics(self,
                                     scores: np.ndarray,
                                     relevance_threshold: float=0.35,
                                     top_k: int=3) -> Tuple[float, int, float]:
        if len(scores) == 0:
            return 0.0, 0, 0.0

        relevant_count = int(np.sum(scores >= relevance_threshold))
        relevance_percentage = (relevant_count / len(scores)) * 100
        overall_score = float(np.mean(scores))

        return overall_score, relevant_count, relevance_percentage
    

    def _determine_label(self, relevance_percentage: float, overall_score: float) -> str:
        if relevance_percentage < 30:
            return "not_related"
        elif relevance_percentage >= 70 and overall_score >= 0.65:
            return "highly_related"
        elif relevance_percentage >= 50 and overall_score >= 0.45:
            return "moderately_related"
        else:
            return "partially_related"
        
    
    def score_relevance(self, text: str, topic: str, relevance_threshold: float = 0.35, evidence_count: int = 5) -> Relevance:
        logger.info(f"analyzing relevance for topic: '{topic}'")

        logger.info(f"splitting text into chunks...")
        chunks = self._split_text(text)
        if not chunks:
            return Relevance(0.0, 0.0, "not_related", [], 0, 0, "none")

        logger.info(f"performing bi-encoding scoring...")
        bi_scores = self._bi_encoder_scoring(chunks, topic)

        logger.info(f"calculating relevance...")
        overall_score, relevant_count, relevance_percentage = self._calculate_relevance_metrics(bi_scores, relevance_threshold)
        label = self._determine_label(relevance_percentage, overall_score)

        evidence_indices = np.argsort(bi_scores)[-evidence_count:][::-1]
        evidence = [(float(bi_scores[i]), chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i]) 
                    for i in evidence_indices]

        logger.info(f"analysis complete. overall score: {overall_score:.3f}, relevance: {relevance_percentage:.3f}, label: {label}")
        return Relevance(
            overall_score=overall_score,
            relevance_percentage=relevance_percentage,
            label=label,
            evidence=evidence,
            chunk_count=len(chunks),
            relevance_chunk_count=relevant_count,
            method_used="bi_encoder"
        )