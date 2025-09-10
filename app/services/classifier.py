import logging
from typing import List, Dict, Any
from transformers import pipeline

logger = logging.getLogger(__name__)


class SensitivityClassifier:
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        self.classifier = pipeline("text-classification", model=model_name, top_k=1)
        logger.info(f"classifier model {model_name} loaded successfully.")

    def classifier_score(self, chunks: List[str]) -> List[Dict[str, Any]]:
        try:
            results = self.classifier(chunks)
            output = [
                {
                    "text": chunk,
                    "label": scores[0]["label"],
                    "score": scores[0]["score"],
                }
                for chunk, scores in zip(chunks, results)
            ]
            return output
        except Exception as e:
            logger.error(f"Sensitivity classification failed: {e}")
            return []
        
    def aggregate_scores(self, results: List[Dict[str, Any]], threshold: float=0.4) -> Dict[str, float]:
        if not results:
            return {"sensitivity_score": 0.0, "evidences": None}
        total_score = 0.0
        evidences: Dict[str, float] = {}
        for r in results:
            score = r.get("score", 0.0)
            total_score += score
            if score > threshold:
                evidences[r["text"]] = score
        average_score = (total_score/len(results)) * 100
        return {
            "sensitivity_score": average_score,
            "evidences": evidences if evidences else None,
        }