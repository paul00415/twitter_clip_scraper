# Twitter Clip Scraper with AI Selection

A sophisticated Python tool that scrapes Twitter for video content, analyzes it using advanced AI, and intelligently selects the best continuous clip matching your criteria. Features comprehensive logging to show exactly which candidates were considered and why the final clip was chosen.

## Features

- 🔍 **Smart Twitter Scraping**: Uses twikit to search for video tweets with real API call evidence
- 🤖 **AI-Powered Filtering**: LangChain-based text analysis with LLM reasoning for candidate shortlisting
- 👁️ **Advanced Vision Analysis**: Gemini 2.5 Flash API for detailed video content analysis and speaker identification
- 🎯 **Intelligent Clip Selection**: LangGraph orchestration with AI ranking and confidence scoring
- 📊 **Rich CLI Interface**: Beautiful terminal output with real-time progress tracking
- 🔄 **Structured Output**: Clean JSON/Pydantic responses with detailed confidence scores and reasoning
- 📋 **Comprehensive Logging**: Clear audit trail showing candidates at each step and AI decision-making
- 🏗️ **Modular Architecture**: Clean separation of concerns with dependency injection and abstract interfaces

## Demonstration

<video controls src="video.mp4" title="Title"></video>

## Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Twitter account (optional, for authenticated scraping with higher rate limits)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/paul00415/twitter_clip_scraper.git
cd twitter_clip_scraper
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the template and edit with your values
cp .env.example .env
# Edit .env file with your actual API keys
```

```

### Usage

#### Basic Usage

```bash
python main.py --description "Tesla Cybertruck unveiling" --duration 12
```

#### Advanced Usage

```bash
python main.py \
  --description "Trump talking about Charlie Kirk" \
  --duration 15 \
  --max-candidates 20 \
  --output json \
  --verbose
```

#### Command Line Options

- `--description`: Media description to search for (required)
- `--duration`: Target clip duration in seconds (required)
- `--max-candidates`: Maximum candidates to consider from Twitter (default: 30)
- `--output`: Output format - `rich` (beautiful) or `json` (default: rich)
- `--verbose`: Enable detailed logging to see AI decision-making
- `--version`: Show version information

#### Testing the Installation

```bash
# Run unit tests
python -m pytest tests/ -v
```

#### Seed Search Terms for Testing

Good seed search terms should be specific, likely to have video content, and represent current or popular topics:

**1. `"Trump talking about Charlie Kirk"`**
   - High likelihood of video content from Tesla's official announcements
   - Multiple video formats (speeches, demonstrations, reactions)
   - Good for testing clip selection of specific events

**2. `"Cristiano Ronaldo free kick"`**
   - Sports content with lots of video highlights
   - Multiple angles and durations
   - Good for testing AI's ability to identify key moments in video

**Tips for Good Search Terms:**
- Use current events, product launches, or popular personalities
- Include specific actions (unveiling, celebration, demonstration)
- Avoid overly broad terms that return too many generic results
- Test with both short-form (speeches) and dynamic (sports) content

#### Testing with Local Video Files

For development and testing, you can use local video files instead of downloading from Twitter:

```bash
# Create downloads directory and place test video files
mkdir -p downloads
cp /path/to/your/video.mp4 ./downloads/video.mp4

# The system will detect and analyze local files for testing
python main.py --description "test video analysis" --duration 10
```

This allows you to test the vision analysis pipeline without relying on Twitter API calls.

## Example Output

### Rich Format with Progress Tracking
```
✓ Configuration loaded successfully
⚙️ Configuration
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Setting        ┃ Value          ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Description    │ Tesla Cybertruck unveiling │
│ Duration       │ 12 seconds     │
│ Max Candidates │ 30             │
│ Output Format  │ rich           │
└────────────────┴────────────────┘

📊 SCRAPING COMPLETE: Found 5 tweet candidates
   Candidates: @elonmusk: "Tesla Cybertruck revolution...", @tesla: "New Cybertruck features...", and 3 more

📊 FILTERING COMPLETE: 3/5 candidates passed text filtering
   Passed: @elonmusk: "Tesla Cybertruck revolution...", @tesla: "New Cybertruck features..."

📊 VISION ANALYSIS COMPLETE: Analyzed 3/3 videos with AI vision
   High relevance (≥0.7): 2 videos
   Best: cybertruck_demo.mp4 (score: 0.89)

🎯 CLIP SELECTION COMPLETE: Final clip selected with confidence 0.91
   Chosen: elon_demo by Elon Musk demonstrating Cybertruck features at 25-37s...
   Clip: 25.2s-37.2s (12.0s)
   Alternatives available: 2 backup clips

🎬 Clip Selection Result
┌─────────────────────────────────────────────────────────┐
│ Selected Clip                                           │
│                                                         │
│ Tweet URL: https://x.com/elonmusk/status/123456789      │
│ Video URL: https://video.twimg.com/...                 │
│ Start Time: 25.2s                                       │
│ End Time: 37.2s                                         │
│ Duration: 12.0s                                         │
│ Confidence: 0.91                                        │
│                                                         │
│ Reason: Elon Musk demonstrating Cybertruck features.   │
│ Clear product showcase with good audio quality.        │
└─────────────────────────────────────────────────────────┘
```

### JSON Format
```json
{
  "tweet_url": "https://x.com/user/status/123456789",
  "video_url": "https://video.twimg.com/...",
  "start_time_s": 47.2,
  "end_time_s": 59.2,
  "confidence": 0.86,
  "reason": "Speaker identified as Donald Trump. Mentions Charlie Kirk at 49–52s. Continuous speech. Clear audio and face.",
  "alternates": [
    {"start_time_s": 122.1, "end_time_s": 134.1, "confidence": 0.77}
  ],
  "trace": {
    "candidates_considered": 18,
    "filtered_by_text": 9,
    "vision_calls": 6,
    "final_choice_rank": 1
  }
}
```

## Architecture

The system follows a clean modular architecture with dependency injection and comprehensive logging:

```
ai_twitter_scraper/
├── scraper/           # Twitter API integration with real call evidence
├── filters/           # AI-powered text filtering with LLM reasoning
├── vision/            # Gemini vision analysis with speaker detection
├── selector/          # Intelligent clip ranking and selection
├── prompts/           # Structured prompts for consistent AI responses
├── tests/             # Unit tests for filtering and timestamp math
├── pipeline.py        # LangGraph orchestration with progress tracking
├── main.py           # CLI interface with rich output
├── config.py         # Environment variable management
├── env_template.txt  # Environment setup template
└── README.md         # This file
```

### Components

1. **Twitter Scraper** (`scraper/`)
   - Twikit-based Twitter API integration with authentication
   - Real API call logging with request/response evidence
   - Video metadata extraction and candidate ranking
   - Rate limiting and retry logic with exponential backoff

2. **AI Text Filter** (`filters/`)
   - LangChain-powered LLM analysis for relevance filtering
   - Batch processing for efficient candidate evaluation
   - Structured output parsing with confidence scoring
   - Clear logging of which candidates passed/failed and why

3. **Vision Analyzer** (`vision/`)
   - Google Gemini 2.5 Flash integration for video analysis
   - Speaker identification, content summarization, and clip detection
   - Real API call evidence with response size logging
   - Fallback handling for API failures

4. **Clip Selector** (`selector/`)
   - AI-powered ranking using structured LLM prompts
   - Confidence-based selection with detailed reasoning
   - Alternative clip generation and backup options
   - Clear explanation of final choice criteria

5. **Pipeline Orchestration** (`pipeline.py`)
   - LangGraph state management with dependency injection
   - Comprehensive progress logging showing candidates at each step
   - Error handling with graceful degradation
   - Real-time progress updates via callbacks

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required. Your Gemini API key
- `TWITTER_USERNAME`: Optional. Twitter username for authentication
- `TWITTER_PASSWORD`: Optional. Twitter password for authentication

### Settings

Modify `config.py` to adjust:
- Rate limiting parameters
- Vision analysis settings
- Clip tolerance values
- Maximum video duration

## Testing

### Running Tests

```bash
# Run comprehensive installation verification
python test_installation.py

# Run unit tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_filtering.py -v      # Test filtering logic
python -m pytest tests/test_timestamp_math.py -v # Test timestamp calculations
```

### Test Coverage

- **Filtering Logic** (4 tests): AI-powered candidate filtering, keyword extraction, relevance scoring, criteria validation
- **Timestamp Math** (5 tests): Duration calculations, tolerance matching, precision handling, clip validation
- **Data Models**: Pydantic validation, structured output parsing, edge cases
- **Integration**: Pipeline orchestration, dependency injection, error handling

All **9 tests passing** ✅ with comprehensive coverage of core functionality.

## Error Handling

The system includes comprehensive error handling:

- **Rate Limiting**: Automatic retries with exponential backoff
- **Authentication**: Fallback to guest mode if credentials fail
- **API Failures**: Graceful degradation with error reporting
- **Malformed Data**: JSON parsing with fallback responses

## Performance Considerations

- **Concurrent Processing**: Parallel video analysis (limited to 3 concurrent)
- **Caching**: LLM responses can be cached for repeated queries
- **Filtering**: Early text filtering reduces expensive vision calls
- **Rate Limiting**: Respects Twitter and Gemini API limits

## Understanding the Logs

The system provides comprehensive logging to show exactly what happens at each step:

### Log Structure
```
📊 SCRAPING COMPLETE: Found X tweet candidates
   Candidates: @user1: "description...", @user2: "description..."

📊 FILTERING COMPLETE: Y/Z candidates passed text filtering
   Passed: @user1: "description...", @user2: "description..."

📊 VISION ANALYSIS COMPLETE: Analyzed A/B videos with AI vision
   High relevance (≥0.7): C videos

🎯 CLIP SELECTION COMPLETE: Final clip selected with confidence 0.X
   Chosen: video_id by AI reasoning...
   Clip: start-end (duration)
```

### What the Logs Tell You
- **Real API Calls**: Evidence of actual Twitter and Gemini API usage
- **Candidate Tracking**: Which tweets/videos were considered at each step
- **AI Reasoning**: Why candidates were accepted/rejected
- **Final Choice**: Detailed explanation of the selected clip

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: Google API key not configured!
   ```
   Solution: Set `GOOGLE_API_KEY` environment variable or create `.env` file

2. **Authentication Failed**
   ```
   Authentication failed: Invalid credentials
   ```
   Solution: Check Twitter credentials or continue with guest mode (limited functionality)

3. **No Candidates Found**
   ```
   📊 SCRAPING COMPLETE: Found 0 tweet candidates
   ```
   Solution: Try different search terms, current events, or popular topics

4. **Rate Limit Exceeded**
   ```
   Rate limit exceeded, using fallback analysis
   ```
   Solution: Wait and retry, reduce `--max-candidates`, or use authenticated mode

### Debug Mode

Enable detailed logging to see AI decision-making:

```bash
python main.py --description "your query" --duration 12 --verbose
```

This shows internal LLM responses, API call details, and step-by-step reasoning.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangGraph**: Workflow orchestration with state management
- **LangChain**: LLM integration and structured output parsing
- **Google Gemini 2.5 Flash**: Advanced vision analysis and speaker detection
- **Twikit**: Twitter API integration with authentication support
- **Pydantic**: Data validation and structured models
- **Rich**: Beautiful terminal interfaces and progress tracking
- **Tenacity**: Robust retry logic and rate limiting

## Key Technologies

- **Python 3.12+** with modern async/await patterns
- **Modular architecture** with dependency injection and abstract interfaces
- **Comprehensive logging** for debugging and monitoring
- **Structured AI prompts** for consistent LLM responses
- **Real API call evidence** throughout the pipeline
