"""Text-based filtering for tweet candidates using LangChain."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from filters.models import (
    FilterResult,
    FilterCriteria,
    RelevanceScore,
    FilteringStrategy,
    ITextFilter,
    TweetRelevanceOutput,
)
from scraper.models import TweetCandidate
from config import settings
from prompts.filtering_prompts import BATCH_FILTERING_PROMPT

logger = logging.getLogger(__name__)


class TextFilter(ITextFilter):
    """Text-based filtering system for tweet candidates."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup LangChain prompts for filtering."""
        # No prompts needed since we're using batch LLM calls directly
        pass

    async def filter_candidates(
        self,
        candidates: List[TweetCandidate],
        description: str,
        duration_seconds: int,
        criteria: Optional[FilterCriteria] = None,
    ) -> List[FilterResult]:
        """Filter tweet candidates based on text analysis."""

        if not criteria:
            criteria = FilterCriteria()

        logger.info(f"🔍 TEXT FILTERING: Analyzing {len(candidates)} candidates for query: '{description}'")

        # Single pass: Batch LLM filtering
        filtered_candidates = await self._quick_filter(candidates, description)
        logger.info(f"Batch filter: {len(filtered_candidates)} candidates passed")

        # Convert candidates to FilterResult format
        results = []
        for candidate in filtered_candidates:
            # Calculate basic relevance score based on keyword matching
            relevance_score = self._calculate_simple_relevance_score(candidate, description)
            
            result = FilterResult(
                passed=True,
                relevance_score=relevance_score,
                reasoning=f"Batch LLM filtering - relevant to query: {description}",
                matched_keywords=self._extract_keywords(candidate.tweet_text, description),
                metadata={"filter_method": "batch_llm"}
            )
            results.append(result)

        logger.info(f"Final filtering: {len(results)} candidates passed")
        return results

    async def _quick_filter(
        self, candidates: List[TweetCandidate], description: str
    ) -> List[TweetCandidate]:
        """Quick relevance check using batch LLM call to avoid rate limits."""

        if not candidates:
            return []

        try:
            # Create batch prompt with all candidates
            candidates_text = ""
            for i, candidate in enumerate(candidates):
                candidates_text += f"{i+1}. Tweet: {candidate.tweet_text[:300]}...\n"

            # Use the batch filtering prompt template
            batch_prompt = BATCH_FILTERING_PROMPT.format(
                description=description,
                candidates_text=candidates_text
            )

            # Single LLM call with structured output for all candidates
            logger.debug(f"Batch prompt: '{batch_prompt[:200]}...'")

            structured_llm = self.llm.with_structured_output(TweetRelevanceOutput)

            try:
                response = await structured_llm.ainvoke(batch_prompt)
                logger.debug(f"Structured LLM response: {response}")

                relevant_indices = response.relevant_indices
                logger.info(f"🤖 LLM FILTERING: {len(relevant_indices)}/{len(candidates)} candidates deemed relevant by AI")
                if relevant_indices:
                    logger.info("   Relevant candidates:")
                    for idx in relevant_indices[:3]:
                        if idx <= len(candidates):
                            candidate = candidates[idx-1]  # indices are 1-based
                            logger.info(f"     ✓ @{candidate.author_handle}: '{candidate.tweet_text[:40]}...'")
                    if len(relevant_indices) > 3:
                        logger.info(f"     ... and {len(relevant_indices)-3} more")

                # Filter candidates based on indices
                filtered = []
                for i, candidate in enumerate(candidates):
                    if (i + 1) in relevant_indices:  # +1 because we numbered from 1
                        logger.debug(f"✅ Tweet {i+1} is relevant: {candidate.tweet_text[:50]}...")
                        filtered.append(candidate)
                    else:
                        logger.debug(f"❌ Tweet {i+1} is not relevant: {candidate.tweet_text[:50]}...")

                return filtered

            except Exception as e:
                logger.warning(f"Failed to get structured LLM response: {e}")
                # If parsing fails, include all candidates to be safe
                return candidates
                
        except Exception as e:
            logger.warning(f"Batch LLM call failed: {e}")
            # If LLM fails, include all candidates to be safe
            return candidates
    
    def _calculate_simple_relevance_score(self, candidate: TweetCandidate, description: str) -> RelevanceScore:
        """Calculate a simple relevance score based on keyword matching and engagement."""
        
        # Extract keywords from description
        desc_keywords = set(re.findall(r'\b\w+\b', description.lower()))
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'about', 'talking', 'saying', 'discussing'
        }
        desc_keywords = desc_keywords - stop_words
        
        # Calculate keyword match
        tweet_words = set(re.findall(r'\b\w+\b', candidate.tweet_text.lower()))
        keyword_match = len(desc_keywords.intersection(tweet_words)) / max(len(desc_keywords), 1)
        
        # Calculate engagement factor
        total_engagement = sum(candidate.engagement_metrics.values())
        engagement_factor = min(total_engagement / 100.0, 1.0)
        
        # Calculate context relevance
        context_relevance = self._calculate_context_relevance(candidate.tweet_text, description)
        
        # Calculate recency factor
        recency_factor = self._calculate_recency_factor(candidate.created_time)
        
        # Overall score
        overall_score = (keyword_match * 0.4 + context_relevance * 0.4 + engagement_factor * 0.1 + recency_factor * 0.1)
        
        return RelevanceScore(
            overall_score=overall_score,
            keyword_match=keyword_match,
            context_relevance=context_relevance,
            engagement_factor=engagement_factor,
            recency_factor=recency_factor
        )
    
    def _extract_keywords(self, tweet_text: str, description: str) -> List[str]:
        """Extract matched keywords from tweet text."""
        
        # Extract keywords from description
        desc_keywords = set(re.findall(r'\b\w+\b', description.lower()))
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'about', 'talking', 'saying', 'discussing'
        }
        desc_keywords = desc_keywords - stop_words
        
        # Find matches in tweet
        tweet_words = set(re.findall(r'\b\w+\b', tweet_text.lower()))
        matched_keywords = list(desc_keywords.intersection(tweet_words))
        
        return matched_keywords



    def _calculate_context_relevance(self, tweet_text: str, description: str) -> float:
        """Calculate context relevance based on semantic similarity to query."""
        
        # Extract meaningful words from description (remove stop words)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "about", "talking", "saying", "discussing", "video"
        }
        
        desc_words = set(re.findall(r'\b\w+\b', description.lower()))
        desc_words = desc_words - stop_words
        
        tweet_words = set(re.findall(r'\b\w+\b', tweet_text.lower()))
        
        # Direct keyword matches
        direct_matches = len(desc_words.intersection(tweet_words))
        
        # Semantic similarity (simple word stemming and synonyms)
        semantic_matches = 0
        semantic_map = {
            "trump": ["donald", "president", "potus"],
            "biden": ["joe", "president", "potus"],
            "charlie": ["kirk"],
            "kirk": ["charlie"],
            "politics": ["political", "government", "policy"],
            "economy": ["economic", "financial", "money"],
            "healthcare": ["health", "medical", "insurance"],
            "immigration": ["immigrant", "border", "migration"],
            "climate": ["environment", "global warming", "green"],
            "education": ["school", "college", "university", "student"]
        }
        
        for desc_word in desc_words:
            # Check for semantic matches
            if desc_word in semantic_map:
                synonyms = semantic_map[desc_word]
                if any(syn in tweet_words for syn in synonyms):
                    semantic_matches += 1
        
        # Calculate relevance score
        total_possible = len(desc_words)
        if total_possible == 0:
            return 0.5  # Neutral score if no meaningful words in description
        
        relevance_score = (direct_matches + semantic_matches * 0.7) / total_possible
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _calculate_recency_factor(self, created_time: datetime) -> float:
        """Calculate recency factor based on tweet age."""
        
        now = datetime.now()
        age_days = (now - created_time).days
        
        if age_days <= 1:
            return 1.0
        elif age_days <= 7:
            return 0.8
        elif age_days <= 30:
            return 0.6
        else:
            return 0.3

