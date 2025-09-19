"""Main pipeline orchestration using LangGraph."""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Callable
from datetime import datetime

from langgraph.graph import StateGraph, END

from scraper.models import SearchQuery, ITwitterScraper
from filters.models import FilterCriteria, ITextFilter
from vision.models import IVideoAnalyzer
from selector.models import SelectionCriteria, SelectionResult, IClipSelector

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State for the LangGraph pipeline."""
    description: str
    duration_seconds: int
    max_candidates: int
    candidates: List[Any]
    filter_results: List[Any]
    vision_results: List[Any]
    final_result: Optional[SelectionResult]
    errors: List[str]
    metadata: Dict[str, Any]


class TwitterClipPipeline:
    """Main pipeline for Twitter clip scraping and selection."""

    def __init__(
        self,
        scraper: ITwitterScraper,
        text_filter: ITextFilter,
        video_analyzer: IVideoAnalyzer,
        clip_selector: IClipSelector
    ):
        self.scraper = scraper
        self.text_filter = text_filter
        self.video_analyzer = video_analyzer
        self.clip_selector = clip_selector

        # Progress callback for dynamic updates
        self._progress_callback = None
        
        # Build the graph
        self.graph = self._build_graph()

    def set_progress_callback(self, callback: Optional[Callable[[str], None]]):
        """Set the progress callback for dynamic updates."""
        self._progress_callback = callback
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("search_tweets", self._search_tweets_node)
        workflow.add_node("filter_candidates", self._filter_candidates_node)
        workflow.add_node("analyze_videos", self._analyze_videos_node)
        workflow.add_node("select_clip", self._select_clip_node)
        
        # Add edges
        workflow.add_edge("search_tweets", "filter_candidates")
        workflow.add_edge("filter_candidates", "analyze_videos")
        workflow.add_edge("analyze_videos", "select_clip")
        workflow.add_edge("select_clip", END)
        
        # Set entry point
        workflow.set_entry_point("search_tweets")
        
        return workflow.compile()
    
    async def _search_tweets_node(self, state: PipelineState) -> PipelineState:
        """Search for tweet candidates."""
        try:
            logger.info(f"Searching for tweets: {state['description']}")

            # Update progress
            if self._progress_callback:
                self._progress_callback("🔍 Searching Twitter...")

            query = SearchQuery(
                description=state['description'],
                duration_seconds=state['duration_seconds'],
                max_candidates=state['max_candidates']
            )

            candidates = await self.scraper.search_tweets(query)
            state['candidates'] = candidates

            logger.info(f"📊 SCRAPING COMPLETE: Found {len(candidates)} tweet candidates")
            if candidates:
                logger.info(f"   Candidates: {', '.join([f'{c.author_handle}: \"{c.tweet_text[:30]}...\"' for c in candidates[:3]])}" +
                           (f" and {len(candidates)-3} more" if len(candidates) > 3 else ""))

        except Exception as e:
            logger.error(f"Tweet search failed: {e}")
            state['errors'].append(f"Search error: {str(e)}")

        return state
    
    async def _filter_candidates_node(self, state: PipelineState) -> PipelineState:
        """Filter candidates based on text analysis."""
        try:
            if not state['candidates']:
                logger.warning("No candidates to filter")
                return state

            logger.info(f"Filtering {len(state['candidates'])} candidates")

            # Update progress
            if self._progress_callback:
                self._progress_callback("📝 Filtering content...")

            criteria = FilterCriteria(
                min_relevance_score=0.1,  # More permissive for testing
                min_engagement_threshold=1  # Lower threshold for testing
            )

            filter_results = await self.text_filter.filter_candidates(
                state['candidates'],
                state['description'],
                state['duration_seconds'],
                criteria
            )

            state['filter_results'] = filter_results

            passed_count = sum(1 for r in filter_results if r.passed)
            logger.info(f"📊 FILTERING COMPLETE: {passed_count}/{len(filter_results)} candidates passed text filtering")
            if passed_count > 0:
                passed_candidates = [state['candidates'][i] for i, r in enumerate(filter_results) if r.passed]
                logger.info(f"   Passed: {', '.join([f'{c.author_handle}: \"{c.tweet_text[:25]}...\"' for c in passed_candidates[:3]])}" +
                           (f" and {len(passed_candidates)-3} more" if len(passed_candidates) > 3 else ""))

        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            state['errors'].append(f"Filtering error: {str(e)}")

        return state
    
    async def _analyze_videos_node(self, state: PipelineState) -> PipelineState:
        """Analyze videos using vision API."""
        try:
            if not state['filter_results']:
                logger.warning("No filtered candidates to analyze")
                return state

            # Get candidates that passed filtering
            filtered_candidates = [
                state['candidates'][i] for i, result in enumerate(state['filter_results'])
                if result.passed
            ]

            if not filtered_candidates:
                logger.warning("No candidates passed filtering")
                return state

            logger.info(f"Analyzing {len(filtered_candidates)} videos")

            # Update progress
            if self._progress_callback:
                self._progress_callback("👁️ Analyzing videos...")

            # Use batch analysis for efficiency
            vision_results = await self.video_analyzer.analyze_videos_batch(
                filtered_candidates,
                state['description'],
                state['duration_seconds']
            )

            state['vision_results'] = vision_results

            analyzed_count = len([r for r in vision_results if r.relevance_score > 0])
            logger.info(f"📊 VISION ANALYSIS COMPLETE: Analyzed {analyzed_count}/{len(vision_results)} videos with AI vision")
            if vision_results:
                high_relevance = [r for r in vision_results if r.relevance_score >= 0.7]
                logger.info(f"   High relevance (≥0.7): {len(high_relevance)} videos")
                if high_relevance:
                    logger.info(f"   Best: {high_relevance[0].video_url.split('/')[-1]} (score: {high_relevance[0].relevance_score:.2f})")

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            state['errors'].append(f"Vision analysis error: {str(e)}")

        return state
    
    async def _select_clip_node(self, state: PipelineState) -> PipelineState:
        """Select the best clip."""
        try:
            if not state['vision_results']:
                logger.warning("No vision results to select from")
                return state

            logger.info("Selecting best clip")

            # Update progress
            if self._progress_callback:
                self._progress_callback("🎯 Selecting best clip...")

            criteria = SelectionCriteria(
                min_confidence=0.1,  # More lenient to allow visioned clips
                max_duration_tolerance=10.0  # More tolerant for duration differences
            )

            # Get filtered candidates for selection
            filtered_candidates = [
                state['candidates'][i] for i, result in enumerate(state['filter_results'])
                if result.passed
            ]

            final_result = await self.clip_selector.select_best_clip(
                filtered_candidates,
                state['filter_results'],
                state['vision_results'],
                state['description'],
                state['duration_seconds'],
                criteria,
                original_candidates_count=len(state['candidates'])
            )

            state['final_result'] = final_result

            logger.info(f"🎯 CLIP SELECTION COMPLETE: Final clip selected with confidence {final_result.confidence:.2f}")
            logger.info(f"   Chosen: {final_result.tweet_url.split('/')[-1]} by {final_result.reason.split('Reason: ')[-1][:60]}...")
            logger.info(f"   Clip: {final_result.start_time_s:.1f}s-{final_result.end_time_s:.1f}s ({final_result.duration_s:.1f}s)")
            if final_result.alternates:
                logger.info(f"   Alternatives available: {len(final_result.alternates)} backup clips")

        except Exception as e:
            logger.error(f"Clip selection failed: {e}")
            state['errors'].append(f"Selection error: {str(e)}")

        return state
    
    async def run(
        self,
        description: str,
        duration_seconds: int,
        max_candidates: int = 30,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SelectionResult:
        """Run the complete pipeline."""

        logger.info(f"Starting pipeline for: {description}")

        # Set progress callback
        self.set_progress_callback(progress_callback)

        # Initialize state
        initial_state: PipelineState = {
            "description": description,
            "duration_seconds": duration_seconds,
            "max_candidates": max_candidates,
            "candidates": [],
            "filter_results": [],
            "vision_results": [],
            "final_result": None,
            "errors": [],
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "pipeline_version": "1.0"
            }
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Add end time to metadata
        final_state["metadata"]["end_time"] = datetime.now().isoformat()
        
        if final_state["final_result"]:
            return final_state["final_result"]
        else:
            # Return empty result if pipeline failed
            from selector.models import SelectionResult, TraceInfo
            return SelectionResult(
                tweet_url="",
                video_url="",
                start_time_s=0.0,
                end_time_s=0.0,
                duration_s=0.0,
                confidence=0.0,
                reason="Pipeline failed to produce result",
                trace=TraceInfo()
            )