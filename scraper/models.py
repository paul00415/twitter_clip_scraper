"""Data models for the scraper module."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel, Field


class VideoVariant(BaseModel):
    """A video variant with specific quality/bitrate."""
    
    content_type: str
    url: str
    bitrate: Optional[int] = None


class VideoInfo(BaseModel):
    """Information about a video attachment."""
    
    url: str
    duration: Optional[float] = None
    aspect_ratio: Optional[List[int]] = None
    duration_millis: Optional[int] = None
    variants: Optional[List[VideoVariant]] = None


class TweetCandidate(BaseModel):
    """A tweet candidate with video attachment."""
    
    tweet_url: str
    tweet_text: str
    author_handle: str
    author_name: str
    created_time: datetime
    engagement_metrics: Dict[str, int] = Field(default_factory=dict)
    video_info: VideoInfo
    raw_tweet_data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Search query parameters."""
    
    description: str
    duration_seconds: int
    max_candidates: int = 30
    language: str = "en"
    
    def to_search_terms(self) -> List[str]:
        """Convert description to search terms."""
        # Basic keyword extraction - can be enhanced with NLP
        terms = self.description.lower().split()
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        return [term for term in terms if term not in stop_words and len(term) > 2]


class ITwitterScraper(ABC):
    """Abstract interface for Twitter scraping functionality."""

    @abstractmethod
    async def search_tweets(self, query: SearchQuery) -> List[TweetCandidate]:
        """Search for tweets matching the query."""
        pass