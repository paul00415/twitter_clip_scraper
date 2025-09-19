"""Prompt templates for filtering operations."""

from langchain.prompts import ChatPromptTemplate

# Batch filtering prompt for strict relevance filtering
BATCH_FILTERING_PROMPT = ChatPromptTemplate.from_template("""
STRICT relevance filtering for video analysis.

Query: {description}

Tweets to check:
{candidates_text}

CRITERIA: Only select tweets that are HIGHLY RELEVANT to the query. Be very selective.

RELEVANCE REQUIREMENTS:
- Must contain DIRECT mentions of key people/topics from the query
- Must be about the SPECIFIC topic, not just tangentially related
- Avoid tweets that are only loosely connected or just mention keywords
- Prioritize tweets with clear, direct relevance over general mentions

SELECTION GUIDELINES:
- Select AT MOST 30% of tweets (be very selective)
- Only include tweets that clearly match the query intent
- Exclude tweets that are just about related topics or people
- Focus on tweets that directly address the query topic

Respond with ONLY a JSON array of numbers for tweets that meet these strict criteria.

Example: [1, 3] (if only tweets 1 and 3 are highly relevant)
Example: [2] (if only tweet 2 is highly relevant)
Example: [] (if no tweets meet the strict criteria)

Be selective - quality over quantity.
""")