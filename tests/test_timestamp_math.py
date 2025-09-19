"""Tests for timestamp mathematics."""

import pytest
from vision.models import ClipSegment, VideoAnalysisResult


class TestTimestampMath:
    """Test cases for timestamp calculations."""
    
    def test_clip_duration_calculation(self):
        """Test clip duration calculation."""
        
        clip = ClipSegment(
            start_time_s=47.2,
            end_time_s=59.2,
            duration_s=12.0,
            confidence=0.86,
            reasoning="Test clip",
            content_description="Test content"
        )
        
        # Test duration calculation
        calculated_duration = clip.end_time_s - clip.start_time_s
        assert abs(calculated_duration - clip.duration_s) < 0.1
        
        # Test duration validation
        assert clip.duration_s > 0
        assert clip.start_time_s >= 0
        assert clip.end_time_s > clip.start_time_s
    
    def test_duration_tolerance_matching(self):
        """Test duration tolerance matching."""
        
        # Create test clips with different durations
        clips = [
            ClipSegment(
                start_time_s=0.0,
                end_time_s=12.0,
                duration_s=12.0,
                confidence=0.8,
                reasoning="Exact match",
                content_description="12 second clip"
            ),
            ClipSegment(
                start_time_s=0.0,
                end_time_s=13.5,
                duration_s=13.5,
                confidence=0.7,
                reasoning="Close match",
                content_description="13.5 second clip"
            ),
            ClipSegment(
                start_time_s=0.0,
                end_time_s=8.0,
                duration_s=8.0,
                confidence=0.6,
                reasoning="Short clip",
                content_description="8 second clip"
            ),
            ClipSegment(
                start_time_s=0.0,
                end_time_s=20.0,
                duration_s=20.0,
                confidence=0.5,
                reasoning="Long clip",
                content_description="20 second clip"
            )
        ]
        
        # Test tolerance matching
        target_duration = 12.0
        tolerance = 2.0
        
        matching_clips = [
            clip for clip in clips 
            if abs(clip.duration_s - target_duration) <= tolerance
        ]
        
        # Should match clips with durations 10.0-14.0 seconds
        assert len(matching_clips) == 2  # 12.0s and 13.5s clips
        
        # Test with stricter tolerance
        strict_tolerance = 1.0
        strict_matching = [
            clip for clip in clips 
            if abs(clip.duration_s - target_duration) <= strict_tolerance
        ]
        
        assert len(strict_matching) == 1  # Only 12.0s clip
    
    def test_clip_segment_validation(self):
        """Test clip segment validation."""
        
        # Valid clip
        valid_clip = ClipSegment(
            start_time_s=10.0,
            end_time_s=22.0,
            duration_s=12.0,
            confidence=0.8,
            reasoning="Valid clip",
            content_description="Valid content"
        )
        
        assert valid_clip.start_time_s >= 0
        assert valid_clip.end_time_s > valid_clip.start_time_s
        assert valid_clip.duration_s > 0
        assert 0.0 <= valid_clip.confidence <= 1.0
        
        # Test invalid clips
        with pytest.raises(ValueError):
            ClipSegment(
                start_time_s=-1.0,  # Invalid negative start time
                end_time_s=10.0,
                duration_s=11.0,
                confidence=0.8,
                reasoning="Invalid clip",
                content_description="Invalid content"
            )
        
        with pytest.raises(ValueError):
            ClipSegment(
                start_time_s=10.0,
                end_time_s=5.0,  # End before start
                duration_s=-5.0,
                confidence=0.8,
                reasoning="Invalid clip",
                content_description="Invalid content"
            )
    
    def test_video_analysis_result_methods(self):
        """Test VideoAnalysisResult methods."""
        
        clips = [
            ClipSegment(
                start_time_s=0.0,
                end_time_s=12.0,
                duration_s=12.0,
                confidence=0.8,
                reasoning="Best clip",
                content_description="Best content"
            ),
            ClipSegment(
                start_time_s=30.0,
                end_time_s=42.0,
                duration_s=12.0,
                confidence=0.7,
                reasoning="Alternate clip",
                content_description="Alternate content"
            ),
            ClipSegment(
                start_time_s=60.0,
                end_time_s=80.0,
                duration_s=20.0,
                confidence=0.6,
                reasoning="Long clip",
                content_description="Long content"
            )
        ]
        
        result = VideoAnalysisResult(
            video_url="https://test.com/video",
            total_duration_s=120.0,
            relevance_score=0.8,
            best_clip=clips[0],
            alternate_clips=clips[1:],
            content_summary="Test video",
            analysis_reasoning="Test analysis"
        )
        
        # Test get_clips_by_duration method
        target_duration = 12.0
        tolerance = 2.0
        
        matching_clips = result.get_clips_by_duration(target_duration, tolerance)
        assert len(matching_clips) == 2  # First two clips match duration
        
        # Test with stricter tolerance
        strict_matching = result.get_clips_by_duration(target_duration, 1.0)
        assert len(strict_matching) == 2  # Still matches first two clips
        
        # Test with very strict tolerance
        very_strict = result.get_clips_by_duration(target_duration, 0.1)
        assert len(very_strict) == 2  # Still matches first two clips
    
    def test_timestamp_precision(self):
        """Test timestamp precision handling."""
        
        # Test with fractional seconds
        clip = ClipSegment(
            start_time_s=47.234567,
            end_time_s=59.234567,
            duration_s=12.0,
            confidence=0.85,
            reasoning="Precise timing",
            content_description="Precise content"
        )
        
        # Should handle fractional seconds correctly
        assert abs(clip.end_time_s - clip.start_time_s - clip.duration_s) < 0.001
        
        # Test rounding for display
        start_rounded = round(clip.start_time_s, 1)
        end_rounded = round(clip.end_time_s, 1)
        
        assert start_rounded == 47.2
        assert end_rounded == 59.2