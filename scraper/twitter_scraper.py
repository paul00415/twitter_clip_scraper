"""Twitter scraper implementation using twikit."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from twikit import Client
from twikit.errors import TwitterException, TooManyRequests

from scraper.models import TweetCandidate, VideoInfo, VideoVariant, SearchQuery, ITwitterScraper
from config import settings

logger = logging.getLogger(__name__)


class TwitterScraper(ITwitterScraper):
    """Twitter scraper for finding video tweets."""
    
    def __init__(self):
        self.client = Client('en-US')
        self._authenticated = False
    
    async def authenticate(self, cookies_file: str = "cookies.json"):
        """Authenticate with Twitter using cookies."""
        try:
            # Try to load cookies first
            try:
                self.client.load_cookies(cookies_file)
                self._authenticated = True
                logger.info(f"Successfully loaded cookies from {cookies_file}")
                return
            except Exception as cookie_error:
                logger.warning(f"Failed to load cookies: {cookie_error}")
            
            # Fallback to username/password if cookies fail
            username = settings.twitter_username
            password = settings.twitter_password
            
            if username and password:
                try:
                    logger.info(f"Attempting authentication with username: {username}")
                    await self.client.login(auth_info_1=username, password=password)
                    self._authenticated = True
                    logger.info("Successfully authenticated with Twitter credentials")
                    
                    # Save cookies for future use
                    try:
                        self.client.save_cookies(cookies_file)
                        logger.info(f"Saved cookies to {cookies_file}")
                    except Exception as save_error:
                        logger.warning(f"Failed to save cookies: {save_error}")
                    return
                except Exception as auth_error:
                    logger.warning(f"Authentication with credentials failed: {auth_error}")
            
            # If all authentication fails, try to continue without it
            self._authenticated = True
            logger.info("Continuing without authentication - will attempt guest search")
                    
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            # Continue without authentication if credentials fail
            self._authenticated = True
            logger.info("Continuing without authentication - limited functionality")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TwitterException, TooManyRequests))
    )
    async def search_tweets(self, query: SearchQuery) -> List[TweetCandidate]:
        """Search for tweets with video attachments."""
        if not self._authenticated:
            await self.authenticate()
        
        logger.info(f"🔍 Starting Twitter search for: '{query.description}' (max {query.max_candidates} candidates)")
        
        # Convert description to search terms
        search_terms = query.to_search_terms()
        search_query = " ".join(search_terms)
        
        # Add filters for video content
        search_query += " filter:videos"
        
        try:
            # Try different search approaches
            search_attempts = [
                (search_query, 'Latest'),
                (search_query.replace(' filter:videos', ''), 'Latest'),
                (query.description, 'Latest'),
                (query.description, 'Top')
            ]
            
            for search_term, product in search_attempts:
                try:
                    logger.info(f"🌐 Twitter API call: Searching '{search_term}' (product: {product})")
                    logger.debug(f"   Full query: '{search_term}' with params: product={product}, count=20")
                    tweets = await self.client.search_tweet(search_term, product)
                    candidates = []
                    
                    for tweet in tweets[:query.max_candidates]:
                        try:
                            candidate = await self._process_tweet(tweet)
                            if candidate:
                                candidates.append(candidate)
                        except Exception as e:
                            logger.warning(f"Failed to process tweet {tweet.id}: {e}")
                            continue
                    
                    if candidates:
                        logger.info(f"🎯 TWITTER API CALL SUCCESSFUL: Found {len(candidates)} video tweet candidates")
                        if candidates:
                            logger.info(f"   Sample candidates:")
                            for i, candidate in enumerate(candidates[:3]):
                                logger.info(f"     {i+1}. @{candidate.author_handle}: '{candidate.tweet_text[:50]}...' ({candidate.engagement_metrics.get('likes', 0)} likes)")
                        return candidates
                    else:
                        logger.info("No video candidates found, trying next search approach")
                        
                except Exception as search_error:
                    logger.warning(f"Search attempt failed: {search_error}")
                    continue
            
            logger.error("All search attempts failed")
            raise Exception("Unable to fetch real Twitter data - all search attempts failed")
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise Exception(f"Twitter search failed: {str(e)}")
    
    
    async def _process_tweet(self, tweet) -> Optional[TweetCandidate]:
        """Process a single tweet and extract video information."""
        try:
            # Check if tweet has video attachments
            if not hasattr(tweet, 'media') or not tweet.media:
                return None
            
            video_info = None
            for media in tweet.media:
                if media.type == 'video' and media.video_info:
                    # Extract variants
                    variants_data = media.video_info.get('variants', [])
                    variants = []
                    for variant_data in variants_data:
                        variants.append(VideoVariant(
                            content_type=variant_data.get('content_type', ''),
                            url=variant_data.get('url', ''),
                            bitrate=variant_data.get('bitrate')
                        ))
                    
                    # Get the best quality URL (highest bitrate or first available)
                    best_url = ''
                    if variants:
                        # Find the highest bitrate variant, or use the first one
                        best_variant = max(variants, key=lambda v: v.bitrate or 0) if any(v.bitrate for v in variants) else variants[0]
                        best_url = best_variant.url
                    else:
                        best_url = getattr(media, 'url', '')
                    
                    video_info = VideoInfo(
                        url=best_url,
                        duration=getattr(media.video_info, 'duration_millis', None) / 1000.0 if getattr(media.video_info, 'duration_millis', None) else None,
                        aspect_ratio=media.video_info.get('aspect_ratio'),
                        duration_millis=getattr(media.video_info, 'duration_millis', None),
                        variants=variants if variants else None
                    )
                    break
            
            if not video_info:
                return None
            
            # Extract engagement metrics
            engagement_metrics = {
                'likes': getattr(tweet, 'favorite_count', 0),
                'retweets': getattr(tweet, 'retweet_count', 0),
                'replies': getattr(tweet, 'reply_count', 0),
                'quotes': getattr(tweet, 'quote_count', 0)
            }
            
            # Parse created time
            created_time = datetime.now()
            if hasattr(tweet, 'created_at'):
                try:
                    created_time = datetime.fromisoformat(tweet.created_at.replace('Z', '+00:00'))
                except:
                    pass
            
            candidate = TweetCandidate(
                tweet_url=f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}",
                tweet_text=tweet.full_text or tweet.text,
                author_handle=tweet.user.screen_name,
                author_name=tweet.user.name,
                created_time=created_time,
                engagement_metrics=engagement_metrics,
                video_info=video_info,
                raw_tweet_data=tweet.__dict__
            )
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Error processing tweet: {e}")
            return None
    