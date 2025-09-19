"""Configuration settings for the Twitter Clip Scraper."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    google_api_key: str  # Required: set GOOGLE_API_KEY environment variable
    twitter_username: Optional[str] = None  # Optional: set TWITTER_USERNAME environment variable
    twitter_password: Optional[str] = None  # Optional: set TWITTER_PASSWORD environment variable
    
    # Scraping settings
    max_candidates: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 2.0
    
    # Vision analysis settings
    max_video_duration: int = 600  # 10 minutes max
    clip_tolerance: float = 2.0  # ±2 seconds tolerance
    
    # API rate limiting settings
    api_rate_limit_delay: float = 3.0  # seconds between API calls
    max_concurrent_requests: int = 1  # max concurrent API requests
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "" 


# Global settings instance
settings = Settings()