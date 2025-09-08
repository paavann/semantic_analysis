import logging
import numpy as np
from app.models.relevance_model import Relevance
from app.services.tokenizer import split_text
from app.services.biencoder import BiEncoder
from app.services.calculate_relevance import calculate_relevance_metrics, determine_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TopicRelevanceScorer:
    def __init__(self,
                 bi_encoder_model: str="all-roberta-large-v1",
                 max_chunk_chars: int=550,):
        
        self.max_chunk_chars = max_chunk_chars
        self.scorer = BiEncoder(bi_encoder_model)
    
    def score_relevance(self,
                        text: str,
                        topic: str,
                        relevance_threshold: float = 0.15,
                        evidence_count: int = 5) -> Relevance:
        logger.info(f"analyzing relevance for topic: {topic}")

        logger.info(f"splitting text into chunks...")
        chunks = split_text(text, self.max_chunk_chars)
        if not chunks:
            return Relevance(0.0, 0.0, "none", [], 0, 0, "none")

        logger.info(f"performing bi-encoding scoring...")
        bi_scores = self.scorer.score(chunks, topic)

        logger.info(f"calculating relevance...")
        overall_score, relevant_count, relevance_percentage, label = calculate_relevance_metrics(bi_scores, relevance_threshold)

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