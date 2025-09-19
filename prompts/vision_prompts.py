"""Prompt templates for vision analysis operations."""

from langchain.prompts import ChatPromptTemplate

# Batch video analysis prompt
BATCH_VIDEO_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
Video relevance analysis for query: "{description}"

Target duration: {duration_seconds} seconds

ANALYSIS CRITERIA:
- Look for videos that contain content related to "{description}"
- Consider both visual and audio content
- Include videos that mention or discuss the topic
- Prioritize videos with clear relevance over completely unrelated content

RELEVANCE REQUIREMENTS:
- Visual or audio content related to the query topic
- Mentions of key people/topics from the query
- Discussion or commentary about the topic
- Any content that could be relevant to the query

SELECTION GUIDELINES:
- Select videos that have reasonable relevance to the query
- Aim to select 40-60% of videos (balanced approach)
- Include videos with moderate to high relevance
- Exclude only videos that are completely unrelated

For SELECTED videos only, provide analysis in this JSON format:
{{
    "selected_videos": [0, 2, 5],
    "video_analyses": [
        {{
            "video_index": 0,
            "relevance_score": 0.85,
            "content_summary": "Brief summary of video content",
            "best_clip": {{
                "start_time_s": 5.0,
                "end_time_s": 17.0,
                "confidence": 0.9,
                "reasoning": "Why this is the best clip"
            }},
            "speakers_detected": [
                {{
                    "name": "Speaker Name",
                    "confidence": 0.9,
                    "appearance_times": [5.2, 12.8]
                }}
            ]
        }}
    ]
}}

Analyze the provided videos for relevance to: "{description}"
Select videos that have reasonable relevance to the query topic.
Respond with ONLY the JSON, no additional text.
""")

# Single video analysis prompt
SINGLE_VIDEO_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
Analyze this video for the following query: "{description}"

Please provide a comprehensive analysis in JSON format with the following structure:

{{
    "relevance_score": 0.85,
    "content_summary": "Brief summary of video content",
    "analysis_reasoning": "Why this video is/isn't relevant to the query",
    "speakers_detected": [
        {{
            "name": "Speaker Name",
            "description": "Description of speaker",
            "confidence": 0.9,
            "appearance_times": [5.2, 12.8, 25.1]
        }}
    ],
    "best_clip": {{
        "start_time_s": 5.0,
        "end_time_s": 17.0,
        "duration_s": 12.0,
        "confidence": 0.9,
        "reasoning": "Why this is the best clip",
        "content_description": "What happens in this clip",
        "audio_quality": "clear|good|poor",
        "visual_quality": "clear|good|poor"
    }},
    "alternate_clips": [
        {{
            "start_time_s": 20.0,
            "end_time_s": 32.0,
            "duration_s": 12.0,
            "confidence": 0.7,
            "reasoning": "Alternative relevant segment",
            "content_description": "What happens in this clip",
            "audio_quality": "clear|good|poor",
            "visual_quality": "clear|good|poor"
        }}
    ]
}}

Focus on:
1. Relevance to the query: "{description}"
2. Speaker identification and timestamps
3. Best segments that match the target duration: {duration_seconds} seconds
4. Audio and visual quality assessment
5. Specific timestamps for key moments

Respond with ONLY the JSON, no additional text.
""")