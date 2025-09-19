"""Data models for the vision analysis module."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SpeakerDetection(BaseModel):
    """Speaker detection in video analysis."""
    name: Optional[str] = None
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    appearance_times: List[float] = Field(default_factory=list)


class ClipAnalysis(BaseModel):
    """Best clip analysis for a video."""
    start_time_s: float = Field(ge=0.0)
    end_time_s: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class VideoAnalysis(BaseModel):
    """Analysis for a single video."""
    video_index: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    content_summary: str
    best_clip: ClipAnalysis
    speakers_detected: List[SpeakerDetection] = Field(default_factory=list)


class BatchVideoAnalysisOutput(BaseModel):
    """Structured output for batch video analysis."""
    selected_videos: List[int] = Field(description="Indices of selected videos")
    video_analyses: List[VideoAnalysis] = Field(description="Detailed analyses for selected videos", default_factory=list)


class SpeakerInfo(BaseModel):
    """Information about a speaker in the video."""
    
    name: Optional[str] = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    appearance_times: List[float] = Field(default_factory=list)


class ClipSegment(BaseModel):
    """A continuous clip segment from the video."""
    
    start_time_s: float = Field(ge=0.0)
    end_time_s: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    speaker_info: Optional[SpeakerInfo] = None
    content_description: str
    audio_quality: str = "unknown"
    visual_quality: str = "unknown"


class VideoAnalysisResult(BaseModel):
    """Result of video analysis."""
    
    video_url: str
    total_duration_s: float
    relevance_score: float = Field(ge=0.0, le=1.0)
    best_clip: Optional[ClipSegment] = None
    alternate_clips: List[ClipSegment] = Field(default_factory=list)
    speakers_detected: List[SpeakerInfo] = Field(default_factory=list)
    content_summary: str
    analysis_reasoning: str
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_clips_by_duration(self, target_duration: float, tolerance: float = 2.0) -> List[ClipSegment]:
        """Get clips that match the target duration within tolerance."""
        all_clips = [self.best_clip] + self.alternate_clips if self.best_clip else self.alternate_clips
        return [
            clip for clip in all_clips
            if abs(clip.duration_s - target_duration) <= tolerance
        ]


class IVideoAnalyzer(ABC):
    """Abstract interface for video analysis functionality."""

    @abstractmethod
    async def analyze_videos_batch(
        self,
        candidates: List["TweetCandidate"],
        description: str,
        duration_seconds: int
    ) -> List[VideoAnalysisResult]:
        """Analyze multiple video candidates in batch."""
        pass