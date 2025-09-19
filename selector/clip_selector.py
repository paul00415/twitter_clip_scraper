"""Clip selection and ranking system."""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from selector.models import SelectionResult, SelectionCriteria, TraceInfo, IClipSelector, AlternateRanking, ClipSelectionOutput
from scraper.models import TweetCandidate
from filters.models import FilterResult
from vision.models import VideoAnalysisResult
from config import settings
from prompts.selector_prompts import CLIP_SELECTION_PROMPT

logger = logging.getLogger(__name__)


class ClipSelector(IClipSelector):
    """Clip selection and ranking system."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompts for final selection."""

        self.selection_prompt = CLIP_SELECTION_PROMPT

    async def select_best_clip(
        self,
        candidates: List[TweetCandidate],
        filter_results: List[FilterResult],
        vision_results: List[VideoAnalysisResult],
        description: str,
        duration_seconds: int,
        criteria: Optional[SelectionCriteria] = None,
        original_candidates_count: Optional[int] = None,
    ) -> SelectionResult:
        """Select the best clip from analyzed candidates."""

        start_time = time.time()

        if not criteria:
            criteria = SelectionCriteria()

        logger.info(
            f"Selecting best clip from {len(vision_results)} analyzed candidates"
        )

        # Create trace info
        trace = TraceInfo(
            candidates_considered=original_candidates_count or len(candidates),
            filtered_by_text=len(filter_results),
            vision_calls=len(vision_results),
            processing_time_s=0.0,
        )

        try:
            # Filter results that meet basic criteria
            valid_results = self._filter_by_criteria(vision_results, criteria, duration_seconds)

            if not valid_results:
                logger.warning("No candidates meet the selection criteria, using all available candidates")
                # Use all vision results if none meet criteria - pick the best available
                valid_results = vision_results
                if not valid_results:
                    logger.warning("No vision results available")
                    return self._create_fallback_result(
                        candidates[0] if candidates else None, trace
                    )

            # Rank candidates using LLM
            ranked_results = await self._rank_candidates(
                valid_results, description, duration_seconds
            )

            # Select the best one
            best_result = ranked_results[0]
            alternates = ranked_results[1:3]  # Top 2 alternates

            # Create final result
            result = self._create_selection_result(
                best_result, alternates, candidates, trace, description
            )

            trace.processing_time_s = time.time() - start_time
            result.trace = trace

            logger.info(f"🎯 CLIP SELECTION: Final clip selected with confidence {result.confidence:.2f}")
            logger.info(f"   Chosen video: {result.video_url.split('/')[-1]}")
            logger.info(f"   Reasoning: {result.reason}")
            if result.alternates:
                logger.info(f"   Backup options: {len(result.alternates)} alternative clips available")
            return result

        except Exception as e:
            logger.error(f"Selection failed: {e}")
            trace.errors_encountered.append(str(e))
            trace.processing_time_s = time.time() - start_time

            return self._create_fallback_result(
                candidates[0] if candidates else None, trace, str(e)
            )

    def _filter_by_criteria(
        self, vision_results: List[VideoAnalysisResult], criteria: SelectionCriteria, target_duration: float = 15.0
    ) -> List[VideoAnalysisResult]:
        """Filter results by selection criteria."""

        valid_results = []

        for result in vision_results:
            if not result.best_clip:
                logger.debug("Skipping result: no best_clip")
                continue

            clip = result.best_clip
            logger.debug(f"Checking clip: confidence={clip.confidence}, duration={clip.duration_s}, audio={clip.audio_quality}, visual={clip.visual_quality}")

            # Check confidence
            if clip.confidence < criteria.min_confidence:
                logger.debug(f"Skipping result: confidence {clip.confidence} < {criteria.min_confidence}")
                continue

            # Check duration tolerance
            duration_diff = abs(clip.duration_s - target_duration)
            if duration_diff > criteria.max_duration_tolerance:
                logger.debug(f"Skipping result: duration diff {duration_diff} > {criteria.max_duration_tolerance}")
                continue

            # Check audio quality
            if criteria.min_audio_quality == "clear" and clip.audio_quality not in [
                "clear",
                "good",
            ]:
                logger.debug(f"Skipping result: audio quality {clip.audio_quality} not in ['clear', 'good']")
                continue

            # Check visual quality
            if criteria.min_visual_quality == "good" and clip.visual_quality not in [
                "good",
                "excellent",
            ]:
                logger.debug(f"Skipping result: visual quality {clip.visual_quality} not in ['good', 'excellent']")
                continue

            logger.debug("Result passed all criteria")
            valid_results.append(result)

        return valid_results

    async def _rank_candidates(
        self,
        vision_results: List[VideoAnalysisResult],
        description: str,
        duration_seconds: int,
    ) -> List[VideoAnalysisResult]:
        """Rank candidates using LLM analysis."""

        # Prepare analysis data for LLM
        candidates_analysis = []
        for i, result in enumerate(vision_results):
            clip = result.best_clip
            analysis_data = {
                "index": i,
                "video_url": result.video_url,
                "relevance_score": result.relevance_score,
                "clip_confidence": clip.confidence,
                "duration_match": abs(clip.duration_s - duration_seconds),
                "speaker_info": (
                    clip.speaker_info.name if clip.speaker_info else "Unknown"
                ),
                "content_description": clip.content_description,
                "audio_quality": clip.audio_quality,
                "visual_quality": clip.visual_quality,
                "reasoning": clip.reasoning,
            }
            candidates_analysis.append(analysis_data)

        # Get LLM ranking with structured output
        structured_llm = self.llm.with_structured_output(ClipSelectionOutput)

        try:
            # Format the prompt properly
            prompt_text = self.selection_prompt.format(
                description=description,
                duration_seconds=duration_seconds,
                candidates_analysis=str(candidates_analysis)
            )

            ranking_data = await structured_llm.ainvoke(prompt_text)

            logger.info(f"🤖 AI RANKING: LLM evaluated {len(vision_results)} candidates and selected index {ranking_data.selected_candidate_index}")
            logger.debug(f"Structured LLM ranking response: {ranking_data}")

            # Extract data from structured response
            selected_index = ranking_data.selected_candidate_index
            alternate_indices = [
                alt.index for alt in ranking_data.alternate_rankings
            ]

            # Reorder results
            ranked_results = []

            # Add selected result first
            if selected_index < len(vision_results):
                ranked_results.append(vision_results[selected_index])

            # Add alternates
            for alt_index in alternate_indices:
                if alt_index < len(vision_results) and alt_index != selected_index:
                    ranked_results.append(vision_results[alt_index])

            # Add remaining results
            for i, result in enumerate(vision_results):
                if i not in [selected_index] + alternate_indices:
                    ranked_results.append(result)

            return ranked_results

        except Exception as e:
            logger.warning(f"Failed to get structured LLM ranking: {e}, using default ranking")
            # Sort by relevance score as fallback
            return sorted(vision_results, key=lambda x: x.relevance_score, reverse=True)

    def _create_selection_result(
        self,
        best_result: VideoAnalysisResult,
        alternates: List[VideoAnalysisResult],
        candidates: List[TweetCandidate],
        trace: TraceInfo,
        description: str,
    ) -> SelectionResult:
        """Create the final selection result."""

        best_clip = best_result.best_clip

        # Create alternate entries
        alternate_entries = []
        for alt_result in alternates:
            if alt_result.best_clip:
                alternate_entries.append(
                    {
                        "start_time_s": alt_result.best_clip.start_time_s,
                        "end_time_s": alt_result.best_clip.end_time_s,
                        "confidence": alt_result.best_clip.confidence,
                    }
                )

        # Find the matching candidate to get the tweet URL
        tweet_url = ""
        for candidate in candidates:
            if candidate.video_info.url == best_result.video_url:
                tweet_url = candidate.tweet_url
                break

        # Fallback if no matching candidate found
        if not tweet_url:
            tweet_url = "https://x.com/user/status/unknown"  # Fallback

        return SelectionResult(
            tweet_url=tweet_url,
            video_url=best_result.video_url,
            start_time_s=best_clip.start_time_s,
            end_time_s=best_clip.end_time_s,
            duration_s=best_clip.duration_s,
            confidence=best_clip.confidence,
            reason=best_clip.reasoning,
            alternates=alternate_entries,
            trace=trace,
        )

    def _create_fallback_result(
        self,
        candidate: Optional[TweetCandidate],
        trace: TraceInfo,
        error: Optional[str] = None,
    ) -> SelectionResult:
        """Create a fallback result when selection fails."""

        if candidate:
            return SelectionResult(
                tweet_url=candidate.tweet_url,
                video_url=candidate.video_info.url,
                start_time_s=0.0,
                end_time_s=12.0,
                duration_s=12.0,
                confidence=0.3,
                reason=f"Fallback selection due to: {error or 'no valid candidates'}",
                trace=trace,
            )
        else:
            return SelectionResult(
                tweet_url="",
                video_url="",
                start_time_s=0.0,
                end_time_s=0.0,
                duration_s=0.0,
                confidence=0.0,
                reason="No candidates available",
                trace=trace,
            )
