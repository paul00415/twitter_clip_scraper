"""Prompt templates for clip selection operations."""

from langchain.prompts import ChatPromptTemplate

# Clip selection prompt
CLIP_SELECTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert at selecting the best video clip from multiple candidates.

Query: {description}
Target Duration: {duration_seconds} seconds

Candidates Analysis:
{candidates_analysis}

Please select the best clip and provide your reasoning. Consider:
1. Relevance to the query
2. Audio and visual quality
3. Speaker identification accuracy
4. Duration match
5. Content continuity

Respond in JSON format:
{{
    "selected_candidate_index": 0,
    "confidence": 0.86,
    "reasoning": "Detailed explanation of selection",
    "alternate_rankings": [
        {{"index": 1, "confidence": 0.77, "reason": "Alternative option"}},
        {{"index": 2, "confidence": 0.65, "reason": "Backup option"}}
    ]
}}
""")
