"""Data models for the filtering module."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TweetRelevanceOutput(BaseModel):
    """Structured output for tweet relevance filtering."""
    relevant_indices: List[int] = Field(description="List of tweet indices (1-based) that are relevant to the query")


class RelevanceScore(BaseModel):
    """Relevance scoring for a tweet candidate."""

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall relevance score")
    keyword_match: float = Field(ge=0.0, le=1.0, description="Keyword matching score")
    context_relevance: float = Field(ge=0.0, le=1.0, description="Context relevance score")
    engagement_factor: float = Field(ge=0.0, le=1.0, description="Engagement quality factor")
    recency_factor: float = Field(ge=0.0, le=1.0, description="Recency factor")


class FilterCriteria(BaseModel):
    """Criteria for filtering tweet candidates."""
    
    min_relevance_score: float = 0.3
    min_engagement_threshold: int = 5
    max_age_days: int = 30
    required_keywords: List[str] = Field(default_factory=list)
    excluded_keywords: List[str] = Field(default_factory=list)
    min_tweet_length: int = 10
    max_tweet_length: int = 500


class FilterResult(BaseModel):
    """Result of filtering a tweet candidate."""
    
    passed: bool
    relevance_score: RelevanceScore
    reasoning: str
    matched_keywords: List[str] = Field(default_factory=list)
    excluded_reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FilteringStrategy(str, Enum):
    """Different filtering strategies."""

    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class ITextFilter(ABC):
    """Abstract interface for text filtering functionality."""

    @abstractmethod
    async def filter_candidates(
        self,
        candidates: List["TweetCandidate"],
        description: str,
        duration_seconds: int,
        criteria: Optional[FilterCriteria] = None,
    ) -> List[FilterResult]:
        """Filter tweet candidates based on text analysis."""
        pass