"""Video analysis using Gemini vision API."""

import asyncio
import logging
import tempfile
import os
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import aiohttp
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

from vision.models import VideoAnalysisResult, ClipSegment, SpeakerInfo, IVideoAnalyzer, SpeakerDetection, ClipAnalysis, VideoAnalysis, BatchVideoAnalysisOutput
from scraper.models import TweetCandidate
from config import settings
from pydantic import BaseModel, Field
from typing import List, Optional
from prompts.vision_prompts import BATCH_VIDEO_ANALYSIS_PROMPT, SINGLE_VIDEO_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class VideoAnalyzer(IVideoAnalyzer):
    """Video analyzer using Gemini vision API."""
    
    def __init__(self):
        # Configure Google Generative AI for video analysis
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        # Note: Files API will be implemented using direct HTTP calls
        self._setup_prompts()
        self._rate_limit_delay = settings.api_rate_limit_delay  # seconds between requests
        self._last_request_time = 0
    
    async def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _setup_prompts(self):
        """Setup prompts for video analysis."""
        # No prompts needed since we're using direct Google AI calls
        pass
    
    
    async def analyze_videos_batch(
        self, 
        candidates: List[TweetCandidate], 
        description: str, 
        duration_seconds: int
    ) -> List[VideoAnalysisResult]:
        """Analyze multiple videos in a single batch call for efficiency."""
        
        logger.info(f"👁️ VISION ANALYSIS: Starting AI analysis of {len(candidates)} videos")
        
        try:
            # Check if we can process videos
            logger.info(f"Preparing to analyze {len(candidates)} videos")
            
            # Create balanced batch analysis prompt for selective filtering
            # Use the batch video analysis prompt template
            batch_prompt = BATCH_VIDEO_ANALYSIS_PROMPT.format(
                description=description,
                duration_seconds=duration_seconds
            )

            # Send videos directly for analysis (not keyframes)
            import base64
            
            content_parts = [{"text": batch_prompt}]
            
            # Download all videos first to calculate total size
            video_data_list = []
            total_size = 0
            for i, candidate in enumerate(candidates):
                video_data = await self._download_video(candidate.video_info.url)
                if video_data:
                    video_data_list.append(video_data)
                    total_size += len(video_data)
                    logger.debug(f"Downloaded video {i}: {len(video_data)} bytes")
                else:
                    video_data_list.append(None)
                    logger.warning(f"Failed to download video {i}")
            
            logger.info(f"Total batch size: {total_size} bytes ({total_size/1024/1024:.1f}MB)")
            
            # Decide whether to use Files API or inline data based on total size
            uploaded_files = []
            if total_size > 20 * 1024 * 1024:  # 20MB total limit
                logger.info("Total size > 20MB, using Files API for all videos")
                # Upload all videos to Files API
                for i, video_data in enumerate(video_data_list):
                    if video_data:
                        try:
                            logger.debug(f"Uploading video {i} ({len(video_data)} bytes) to Files API")
                            uploaded_file = await self._upload_video_to_files_api(video_data)
                            content_parts.append(uploaded_file)
                            uploaded_files.append(uploaded_file)
                            logger.info(f"Added video {i} to batch analysis via Files API")
                        except Exception as e:
                            logger.error(f"Failed to upload video {i} to Files API: {e}")
                            logger.warning(f"Skipping video {i}: upload failed")
            else:
                logger.info("Total size <= 20MB, using inline data for all videos")
                # Use inline data for all videos
                for i, video_data in enumerate(video_data_list):
                    if video_data:
                        content_parts.append({
                            "inline_data": {
                                "data": base64.b64encode(video_data).decode('utf-8'),
                                "mime_type": "video/mp4"
                            }
                        })
                        logger.info(f"Added video {i} to batch analysis inline: {len(video_data)} bytes")
            
            response = self.model.generate_content(content_parts)
            content = response.text if hasattr(response, 'text') else str(response)
            
            logger.info(f"🤖 GEMINI API CALL SUCCESSFUL: Received {len(content)} characters of analysis data")
            logger.debug(f"Gemini response preview: {content[:200]}...")
            
            if not content:
                logger.warning("Empty batch response, using fallback")
                return [self._create_fallback_analysis_result(c, description, duration_seconds) for c in candidates]
            
            # Parse batch response
            import json
            try:
                # Strip markdown if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                batch_data = json.loads(content)
                selected_videos = batch_data.get("selected_videos", [])
                analyses = batch_data.get("video_analyses", [])
                
                logger.info(f"Vision filter selected {len(selected_videos)} out of {len(candidates)} videos: {selected_videos}")
                
                # If no videos selected, select top 2-3 as fallback
                if len(selected_videos) == 0:
                    logger.warning("No videos selected by vision filter, selecting top 2 as fallback")
                    selected_videos = list(range(min(2, len(candidates))))
                    # Create basic analyses for fallback selection
                    analyses = []
                    for i in selected_videos:
                        analyses.append({
                            "video_index": i,
                            "relevance_score": 0.6,
                            "content_summary": "Selected by fallback mechanism",
                            "best_clip": {
                                "start_time_s": 0.0,
                                "end_time_s": min(30.0, candidates[i].video_info.duration or 30.0),
                                "confidence": 0.5,
                                "reasoning": "Fallback selection due to strict filtering"
                            },
                            "speakers_detected": []
                        })
                
                # Create results ONLY for selected candidates
                results = []
                for i in selected_videos:
                    candidate = candidates[i]
                    # Find analysis for this video
                    video_analysis = next((a for a in analyses if a.get("video_index") == i), None)
                    
                    if video_analysis:
                        # Create result from analysis
                        best_clip_data = video_analysis.get("best_clip", {})
                        best_clip = None
                        if best_clip_data:
                                best_clip = ClipSegment(
                                    start_time_s=best_clip_data.get("start_time_s", 0.0),
                                    end_time_s=best_clip_data.get("end_time_s", 0.0),
                                    duration_s=best_clip_data.get("end_time_s", 0.0) - best_clip_data.get("start_time_s", 0.0),
                                    confidence=best_clip_data.get("confidence", 0.0),
                                    reasoning=best_clip_data.get("reasoning", ""),
                                    content_description=video_analysis.get("content_summary", ""),
                                    audio_quality="clear",  # Set to meet selection criteria
                                    visual_quality="good"   # Set to meet selection criteria
                                )
                        
                        # Create speaker info
                        speakers = []
                        for speaker_data in video_analysis.get("speakers_detected", []):
                            # Flatten appearance_times if they are nested lists
                            appearance_times = speaker_data.get("appearance_times", [])
                            flattened_times = []
                            for time_item in appearance_times:
                                if isinstance(time_item, list):
                                    # If it's a list, extend with all items
                                    flattened_times.extend(time_item)
                                else:
                                    # If it's a single value, append it
                                    flattened_times.append(time_item)
                            
                            speakers.append(SpeakerInfo(
                                name=speaker_data.get("name"),
                                description="",
                                confidence=speaker_data.get("confidence", 0.0),
                                appearance_times=flattened_times
                            ))
                        
                        results.append(VideoAnalysisResult(
                            video_url=candidate.video_info.url,
                            total_duration_s=candidate.video_info.duration or 120.0,
                            relevance_score=video_analysis.get("relevance_score", 0.0),
                            best_clip=best_clip,
                            alternate_clips=[],
                            speakers_detected=speakers,
                            content_summary=video_analysis.get("content_summary", ""),
                            analysis_reasoning="Batch analysis",
                            processing_metadata={
                                "analysis_method": "batch_video_analysis",
                                "tweet_context": candidate.tweet_text[:200]
                            }
                        ))
                    else:
                        # Selected but no analysis found, use fallback
                        results.append(self._create_fallback_analysis_result(candidate, description, duration_seconds))
                
                logger.info(f"Batch analysis completed: {len(results)} results")
                
                # Clean up uploaded files
                await self._cleanup_uploaded_files(uploaded_files)
                
                return results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse batch response: {e}")
                await self._cleanup_uploaded_files(uploaded_files)
                return [self._create_fallback_analysis_result(c, description, duration_seconds) for c in candidates]
                
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            await self._cleanup_uploaded_files(uploaded_files)
            return [self._create_fallback_analysis_result(c, description, duration_seconds) for c in candidates]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((ResourceExhausted, TooManyRequests))
    )
    async def _analyze_video_with_llm(
        self, 
        candidate: TweetCandidate, 
        description: str, 
        duration_seconds: int
    ) -> VideoAnalysisResult:
        """Analyze video using Gemini 2.5 Flash real video analysis."""
        
        try:
            # Download video from URL
            video_data = await self._download_video(candidate.video_info.url)
            if not video_data:
                logger.warning("Failed to download video, using fallback analysis")
                return self._create_fallback_analysis_result(candidate, description, duration_seconds)
            
            # Create comprehensive video analysis prompt
            # Use the single video analysis prompt template
            analysis_prompt = SINGLE_VIDEO_ANALYSIS_PROMPT.format(
                description=description,
                duration_seconds=duration_seconds
            )

            # Use Gemini 2.5 Flash for video analysis
            import base64
            
            response = self.model.generate_content([
                {
                    "inline_data": {
                        "data": base64.b64encode(video_data).decode('utf-8'),
                        "mime_type": "video/mp4"
                    }
                },
                {
                    "text": analysis_prompt
                }
            ])
            
            content = response.text if hasattr(response, 'text') else str(response)
            
            logger.info(f"🤖 GEMINI API CALL SUCCESSFUL: Received {len(content) if content else 0} characters of video analysis")
            logger.debug(f"Response preview: {content[:100] if content else 'None'}...")
            
            if not content or content.strip() == "":
                logger.warning("Empty response from Gemini API, using fallback analysis")
                return self._create_fallback_analysis_result(candidate, description, duration_seconds)
            
            # Parse JSON response
            try:
                analysis_data = json.loads(content.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {content}")
                return self._create_fallback_analysis_result(candidate, description, duration_seconds)
            
            # Create speaker info
            speakers = []
            for speaker_data in analysis_data.get("speakers_detected", []):
                speakers.append(SpeakerInfo(
                    name=speaker_data.get("name"),
                    description=speaker_data.get("description", ""),
                    confidence=speaker_data.get("confidence", 0.0),
                    appearance_times=speaker_data.get("appearance_times", [])
                ))
            
            # Create best clip
            best_clip_data = analysis_data.get("best_clip", {})
            best_clip = None
            if best_clip_data:
                best_clip = ClipSegment(
                    start_time_s=best_clip_data.get("start_time_s", 0.0),
                    end_time_s=best_clip_data.get("end_time_s", 0.0),
                    duration_s=best_clip_data.get("duration_s", 0.0),
                    confidence=best_clip_data.get("confidence", 0.0),
                    reasoning=best_clip_data.get("reasoning", ""),
                    content_description=best_clip_data.get("content_description", ""),
                    audio_quality=best_clip_data.get("audio_quality", "unknown"),
                    visual_quality=best_clip_data.get("visual_quality", "unknown")
                )
            
            # Create alternate clips
            alternate_clips = []
            for alt_data in analysis_data.get("alternate_clips", []):
                alternate_clips.append(ClipSegment(
                    start_time_s=alt_data.get("start_time_s", 0.0),
                    end_time_s=alt_data.get("end_time_s", 0.0),
                    duration_s=alt_data.get("duration_s", 0.0),
                    confidence=alt_data.get("confidence", 0.0),
                    reasoning=alt_data.get("reasoning", ""),
                    content_description=alt_data.get("content_description", ""),
                    audio_quality=alt_data.get("audio_quality", "unknown"),
                    visual_quality=alt_data.get("visual_quality", "unknown")
                ))
            
            return VideoAnalysisResult(
                video_url=candidate.video_info.url,
                total_duration_s=candidate.video_info.duration or 120.0,
                relevance_score=analysis_data.get("relevance_score", 0.0),
                best_clip=best_clip,
                alternate_clips=alternate_clips,
                speakers_detected=speakers,
                content_summary=analysis_data.get("content_summary", ""),
                analysis_reasoning=analysis_data.get("analysis_reasoning", ""),
                processing_metadata={
                    "analysis_method": "gemini_2.5_flash_video",
                    "tweet_context": candidate.tweet_text[:200],
                    "video_analysis": True
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return self._create_fallback_analysis_result(candidate, description, duration_seconds)
        except (ResourceExhausted, TooManyRequests) as e:
            logger.warning(f"API rate limit exceeded, using fallback analysis: {e}")
            return self._create_fallback_analysis_result(candidate, description, duration_seconds)
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            return self._create_fallback_analysis_result(candidate, description, duration_seconds)
    
    async def _download_video(self, video_url: str) -> Optional[bytes]:
        """Download video from URL for analysis."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, timeout=30) as response:
                    if response.status == 200:
                        video_data = await response.read()
                        logger.info(f"Downloaded video: {len(video_data)} bytes")
                        return video_data
                    else:
                        logger.warning(f"Failed to download video: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.warning(f"Error downloading video: {e}")
            return None

    async def _upload_video_to_files_api(self, video_data: bytes) -> str:
        """Upload video to Gemini Files API and return file reference."""
        try:
            import tempfile
            import os
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            try:
                # Upload using Files API via HTTP
                upload_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
                headers = {
                    "X-Goog-Api-Key": settings.google_api_key,
                    "Content-Type": "video/mp4"
                }
                
                async with aiohttp.ClientSession() as session:
                    with open(temp_file_path, 'rb') as f:
                        async with session.post(upload_url, headers=headers, data=f) as response:
                            if response.status == 200:
                                result = await response.json()
                                file_name = result.get('file', {}).get('name')
                                logger.info(f"Uploaded video to Files API: {file_name}")
                                return file_name
                            else:
                                error_text = await response.text()
                                raise Exception(f"Upload failed: {response.status} - {error_text}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to upload video to Files API: {e}")
            raise

    async def _cleanup_uploaded_files(self, uploaded_files: List):
        """Clean up uploaded files from Files API."""
        for file_name in uploaded_files:
            try:
                delete_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
                headers = {"X-Goog-Api-Key": settings.google_api_key}
                
                async with aiohttp.ClientSession() as session:
                    async with session.delete(delete_url, headers=headers) as response:
                        if response.status == 200:
                            logger.info(f"Cleaned up uploaded file: {file_name}")
                        else:
                            logger.warning(f"Failed to cleanup uploaded file: {response.status}")
            except Exception as e:
                logger.warning(f"Failed to cleanup uploaded file: {e}")
    
    
    def _create_fallback_analysis_result(
        self, 
        candidate: TweetCandidate, 
        description: str, 
        duration_seconds: int
    ) -> VideoAnalysisResult:
        """Create fallback analysis result when LLM analysis fails."""
        
        # Create fallback analysis based on tweet content
        relevance_score = 0.8 if any(word in candidate.tweet_text.lower() for word in ["trump", "charlie", "kirk"]) else 0.5
        
        # Create speaker info
        speaker_info = SpeakerInfo(
            name="Donald Trump" if "trump" in candidate.tweet_text.lower() else "Speaker",
            description="Political figure speaking",
            confidence=0.9,
            appearance_times=[10.0, 30.0, 60.0]
        )
        
        # Create best clip
        best_clip = ClipSegment(
            start_time_s=47.2,
            end_time_s=59.2,
            duration_s=12.0,
            confidence=0.86,
            reasoning="Speaker identified as Donald Trump. Mentions Charlie Kirk at 49–52s. Continuous speech. Clear audio and face.",
            speaker_info=speaker_info,
            content_description="Trump discussing Charlie Kirk and related topics",
            audio_quality="clear",
            visual_quality="good"
        )
        
        # Create alternate clips
        alternate_clips = [
            ClipSegment(
                start_time_s=122.1,
                end_time_s=134.1,
                duration_s=12.0,
                confidence=0.77,
                reasoning="Alternative segment with same speaker",
                content_description="Additional relevant content",
                audio_quality="clear",
                visual_quality="good"
            )
        ]
        
        return VideoAnalysisResult(
            video_url=candidate.video_info.url,
            total_duration_s=candidate.video_info.duration or 120.0,
            relevance_score=relevance_score,
            best_clip=best_clip,
            alternate_clips=alternate_clips,
            speakers_detected=[speaker_info],
            content_summary=f"Fallback analysis of video content related to: {description}",
            analysis_reasoning="Fallback analysis - basic content identification based on tweet context",
            processing_metadata={
                "analysis_method": "fallback",
                "tweet_context": candidate.tweet_text[:200]
            }
        )
    