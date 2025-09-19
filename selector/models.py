"""Data models for the clip selection module."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AlternateRanking(BaseModel):
    """Alternate ranking for clip selection."""
    index: int = Field(description="Index of the alternate candidate")
    confidence: float = Field(description="Confidence score for this alternate")
    reason: str = Field(description="Reason this is a good alternate")


class ClipSelectionOutput(BaseModel):
    """Structured output for clip selection."""
    selected_candidate_index: int = Field(description="Index of the selected candidate")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    reasoning: str = Field(description="Detailed explanation of selection")
    alternate_rankings: List[AlternateRanking] = Field(description="List of alternate rankings", default_factory=list)


class TraceInfo(BaseModel):
    """Tracing information for the selection process."""
    
    candidates_considered: int = 0
    filtered_by_text: int = 0
    vision_calls: int = 0
    final_choice_rank: int = 1
    processing_time_s: float = 0.0
    errors_encountered: List[str] = Field(default_factory=list)


class SelectionCriteria(BaseModel):
    """Criteria for clip selection."""
    
    min_confidence: float = 0.3  # Lowered from 0.7 to 0.3
    max_duration_tolerance: float = 5.0  # Increased from 2.0 to 5.0
    prefer_higher_engagement: bool = True
    prefer_recent_content: bool = True
    require_speaker_match: bool = False  # Made less strict
    min_audio_quality: str = "unknown"  # Made less strict
    min_visual_quality: str = "unknown"  # Made less strict


class SelectionResult(BaseModel):
    """Final selection result."""
    
    tweet_url: str
    video_url: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    confidence: float
    reason: str
    alternates: List[Dict[str, Any]] = Field(default_factory=list)
    trace: TraceInfo
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching the example output."""
        return {
            "tweet_url": self.tweet_url,
            "video_url": self.video_url,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "confidence": self.confidence,
            "reason": self.reason,
            "alternates": self.alternates,
            "trace": {
                "candidates_considered": self.trace.candidates_considered,
                "filtered_by_text": self.trace.filtered_by_text,
                "vision_calls": self.trace.vision_calls,
                "final_choice_rank": self.trace.final_choice_rank
            }
        }


class IClipSelector(ABC):
    """Abstract interface for clip selection functionality."""

    @abstractmethod
    async def select_best_clip(
        self,
        candidates: List["TweetCandidate"],
        filter_results: List["FilterResult"],
        vision_results: List["VideoAnalysisResult"],
        description: str,
        duration_seconds: int,
        criteria: Optional["SelectionCriteria"] = None,
        original_candidates_count: Optional[int] = None,
    ) -> SelectionResult:
        """Select the best clip from analyzed candidates."""
        pass