from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional, Any


class Relevance(BaseModel):
    overall_score: float
    relevance_percentage: float
    label: str
    evidence: List[Tuple[float, str]]
    chunk_count: int
    relevance_chunk_count: int
    method_used: str
    sensitivity: Optional[Dict[str, Any]]