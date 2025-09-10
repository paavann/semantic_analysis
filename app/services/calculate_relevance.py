from typing import Tuple
import numpy as np



def determine_label(relevance_percentage: float, overall_score: float) -> str:
    if(relevance_percentage<30):
        return "irrelevant"
    elif(relevance_percentage>=70 and overall_score>=0.65):
        return "highly_relevant"
    elif(relevance_percentage>=50 and overall_score>=0.20):
        return "moderately_relevant"
    else:
        return "partially_relevant"

def calculate_relevance_metrics(scores: np.ndarray, relevance_threshold: float=0.15) -> Tuple[float, int, float, str]:
    if(len(scores)) == 0:
        return 0.0, 0, 0.0, "none"
    
    relevant_count = int(np.sum(scores>=relevance_threshold))
    relevance_percentage = (relevant_count/len(scores)) * 100
    overall_score = float(np.mean(scores)) * 100
    label = determine_label(relevance_percentage, overall_score)

    return overall_score, relevant_count, relevance_percentage, label