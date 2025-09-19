#!/usr/bin/env python3
"""Main CLI interface for Twitter Clip Scraper."""

import asyncio
import argparse
import json
import logging
import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from pipeline import TwitterClipPipeline
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def setup_environment():
    """Setup environment variables and check configuration."""

    # Check for required API key
    try:
        # This will raise an exception if GOOGLE_API_KEY is not set
        api_key = settings.google_api_key
        if not api_key or api_key.strip() == "":
            raise ValueError("Empty API key")
    except Exception as e:
        console.print("[red]Error: Google API key not configured![/red]")
        console.print("Please set the GOOGLE_API_KEY environment variable or create a .env file")
        console.print("\nMethods to set the API key:")
        console.print("1. Environment variable: export GOOGLE_API_KEY=your_actual_api_key")
        console.print("2. .env file: echo 'GOOGLE_API_KEY=your_actual_api_key' > .env")
        console.print("3. Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    console.print("[green]✓ Configuration loaded successfully[/green]")
    console.print(f"  - Google API Key: {'✓ Set' if settings.google_api_key else '✗ Not set'}")
    console.print(f"  - Twitter Username: {'✓ Set' if settings.twitter_username else '✗ Not set'}")
    console.print(f"  - Twitter Password: {'✓ Set' if settings.twitter_password else '✗ Not set'}")


def display_result(result, output_format: str):
    """Display the result in the specified format."""
    
    if output_format == "json":
        console.print(json.dumps(result.to_dict(), indent=2))
    else:
        # Rich formatted output
        display_rich_result(result)


def display_rich_result(result):
    """Display result using Rich formatting."""
    
    # Main result panel
    result_panel = Panel(
        f"[bold green]Selected Clip[/bold green]\n\n"
        f"[bold]Tweet URL:[/bold] {result.tweet_url}\n"
        f"[bold]Video URL:[/bold] {result.video_url}\n"
        f"[bold]Start Time:[/bold] {result.start_time_s:.1f}s\n"
        f"[bold]End Time:[/bold] {result.end_time_s:.1f}s\n"
        f"[bold]Duration:[/bold] {result.duration_s:.1f}s\n"
        f"[bold]Confidence:[/bold] {result.confidence:.2f}\n\n"
        f"[bold]Reason:[/bold] {result.reason}",
        title="🎬 Clip Selection Result",
        border_style="green"
    )
    
    console.print(result_panel)
    
    # Alternates table
    if result.alternates:
        table = Table(title="🔄 Alternate Options")
        table.add_column("Start Time", style="cyan")
        table.add_column("End Time", style="cyan")
        table.add_column("Confidence", style="green")
        
        for alt in result.alternates:
            table.add_row(
                f"{alt['start_time_s']:.1f}s",
                f"{alt['end_time_s']:.1f}s",
                f"{alt['confidence']:.2f}"
            )
        
        console.print(table)
    
    # Trace information
    trace_panel = Panel(
        f"[bold]Processing Statistics[/bold]\n\n"
        f"Candidates Considered: {result.trace.candidates_considered}\n"
        f"Filtered by Text: {result.trace.filtered_by_text}\n"
        f"Vision Calls: {result.trace.vision_calls}\n"
        f"Final Choice Rank: {result.trace.final_choice_rank}\n"
        f"Processing Time: {result.trace.processing_time_s:.2f}s",
        title="📊 Trace Information",
        border_style="blue"
    )
    
    console.print(trace_panel)


async def run_pipeline(description: str, duration: int, max_candidates: int, output_format: str):
    """Run the complete pipeline with dynamic progress updates."""

    from scraper.twitter_scraper import TwitterScraper
    from filters.text_filter import TextFilter
    from vision.video_analyzer import VideoAnalyzer
    from selector.clip_selector import ClipSelector
    from pipeline import TwitterClipPipeline

    # Initialize components
    scraper = TwitterScraper()
    text_filter = TextFilter()
    video_analyzer = VideoAnalyzer()
    clip_selector = ClipSelector()

    # Create pipeline with injected dependencies (Dependency Inversion Principle)
    pipeline = TwitterClipPipeline(scraper, text_filter, video_analyzer, clip_selector)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("🚀 Starting...", total=None)

        try:
            # Define progress callback for dynamic updates
            def update_progress(message: str):
                progress.update(task, description=message)

            # Run pipeline with progress callback
            result = await pipeline.run(description, duration, max_candidates, progress_callback=update_progress)

            progress.update(task, description="✅ Complete!")

            # Display result
            display_result(result, output_format)

            return result

        except Exception as e:
            progress.update(task, description="❌ Failed!")
            console.print(f"[red]Error: {str(e)}[/red]")
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Twitter Clip Scraper with AI Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --description "Trump talking about Charlie Kirk" --duration 12
  python main.py --description "Biden discussing economy" --duration 15 --max-candidates 20
  python main.py --description "Elon Musk interview" --duration 30 --output json
        """
    )
    
    parser.add_argument(
        "--description",
        required=True,
        help="Media description to search for"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        help="Target duration in seconds"
    )
    
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum number of candidates to consider (default: 30)"
    )
    
    parser.add_argument(
        "--output",
        choices=["rich", "json"],
        default="rich",
        help="Output format (default: rich)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Twitter Clip Scraper v1.0"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment
    setup_environment()
    
    # Validate arguments
    if args.duration <= 0:
        console.print("[red]Error: Duration must be positive[/red]")
        sys.exit(1)
    
    if args.max_candidates <= 0:
        console.print("[red]Error: Max candidates must be positive[/red]")
        sys.exit(1)
    
    # Display configuration
    config_table = Table(title="⚙️ Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Description", args.description)
    config_table.add_row("Duration", f"{args.duration} seconds")
    config_table.add_row("Max Candidates", str(args.max_candidates))
    config_table.add_row("Output Format", args.output)
    
    console.print(config_table)
    console.print()
    
    # Run the pipeline
    try:
        result = asyncio.run(run_pipeline(
            args.description,
            args.duration,
            args.max_candidates,
            args.output
        ))
        
        if result and result.confidence > 0:
            console.print("\n[green]✓ Pipeline completed successfully![/green]")
        else:
            console.print("\n[yellow]⚠ Pipeline completed but no suitable clip found[/yellow]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Pipeline failed: {str(e)}[/red]")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()