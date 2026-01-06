import asyncio
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


def extract_video_id(url):
    """
    Extract YouTube video ID from URL

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    # Standard youtube.com URL
    if "youtube.com" in url:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        return query_params.get("v", [None])[0]

    # Shortened youtu.be URL
    elif "youtu.be" in url:
        parsed = urlparse(url)
        return parsed.path.strip("/")

    # If it's already just the ID
    elif re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    return None


def format_timestamp(seconds):
    """Convert seconds to [HH:MM:SS] format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    else:
        return f"[{minutes:02d}:{secs:02d}]"


async def get_youtube_transcript(video_url, languages=["en", "es"]):
    """
    Get transcript from YouTube using existing subtitles

    Args:
        video_url: YouTube video URL or video ID
        languages: List of language codes to try (in order of preference)

    Returns:
        Tuple of (transcript_text, metadata) or (None, None) if failed
    """
    print(f"📹 Extracting transcript from: {video_url}")

    # Extract video ID
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()

    if not video_id:
        print(f"✗ Invalid YouTube URL: {video_url}")
        return None, None

    # Run in executor (API calls are I/O bound)
    loop = asyncio.get_event_loop()

    def _get_transcript():
        try:
            # Create API instance
            ytt_api = YouTubeTranscriptApi()

            # Fetch transcript using the instance
            fetched_transcript = ytt_api.fetch(video_id, languages=languages)

            # Convert to raw data (list of dicts)
            transcript_data = fetched_transcript.to_raw_data()

            # Format transcript with timestamps
            formatted_lines = []
            full_text = []

            for entry in transcript_data:
                timestamp = format_timestamp(entry["start"])
                text = entry["text"].strip()

                formatted_lines.append(f"{timestamp} {text}")
                full_text.append(text)

            metadata = {
                "video_id": fetched_transcript.video_id,
                "video_url": f"https://www.youtube.com/watch?v={fetched_transcript.video_id}",
                "language": fetched_transcript.language,
                "language_code": fetched_transcript.language_code,
                "transcript_type": "auto-generated"
                if fetched_transcript.is_generated
                else "manual",
                "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "segment_count": len(transcript_data),
            }

            return {
                "formatted": "\n".join(formatted_lines),
                "plain": " ".join(full_text),
                "metadata": metadata,
            }

        except TranscriptsDisabled:
            print(f"✗ Transcripts are disabled for video: {video_id}")
            return None
        except NoTranscriptFound:
            print(f"✗ No transcript found in languages: {languages}")
            return None
        except VideoUnavailable:
            print(f"✗ Video is unavailable: {video_id}")
            return None
        except CouldNotRetrieveTranscript:
            print(f"✗ Could not retrieve transcript for: {video_id}")
            return None
        except Exception as e:
            print(f"✗ Error getting transcript: {e}")
            return None

    # Run in executor
    result = await loop.run_in_executor(None, _get_transcript)

    if result:
        print(
            f"✓ Transcript extracted ({result['metadata']['segment_count']} segments, {result['metadata']['transcript_type']})"
        )

    return result


def generate_filename(video_id):
    """Generate filename based on video ID"""
    return f"youtube_{video_id}.md"


async def video_to_markdown(
    video_url, output_dir="data/videos", languages=["en", "es"]
):
    """
    Convert YouTube video to Markdown transcript

    Args:
        video_url: YouTube video URL
        output_dir: Output directory
        languages: Preferred languages for transcript

    Returns:
        Dict with operation result
    """
    print(f"📹 Processing: {video_url}")

    try:
        # Get transcript
        result = await get_youtube_transcript(video_url, languages)

        if not result:
            return {
                "video_url": video_url,
                "output_path": None,
                "success": False,
                "error": "Could not extract transcript",
            }

        metadata = result["metadata"]

        # Create Markdown content
        markdown_content = f"""---
video_id: {metadata["video_id"]}
video_url: {metadata["video_url"]}
language: {metadata["language"]}
transcript_type: {metadata["transcript_type"]}
extracted_at: {metadata["extracted_at"]}
---

# YouTube Transcript - {metadata["video_id"]}

**Video URL:** [{metadata["video_url"]}]({metadata["video_url"]})
**Language:** {metadata["language"]}
**Type:** {metadata["transcript_type"]}

## Transcript with Timestamps

{result["formatted"]}

## Plain Text

{result["plain"]}
"""

        # Save to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = generate_filename(metadata["video_id"])
        filepath = output_path / filename

        # Write file async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: filepath.write_text(markdown_content, encoding="utf-8")
        )

        print(f"✓ Saved: {filename}")

        return {
            "video_url": video_url,
            "video_id": metadata["video_id"],
            "output_path": str(filepath),
            "success": True,
            "language": metadata["language"],
            "type": metadata["transcript_type"],
        }

    except Exception as e:
        print(f"✗ Error processing video: {e}")
        return {
            "video_url": video_url,
            "output_path": None,
            "success": False,
            "error": str(e),
        }


async def process_videos(
    video_urls, output_dir="output/videos", languages=["en", "es"], max_concurrent=3
):
    """
    Process multiple YouTube videos concurrently

    Args:
        video_urls: List of YouTube video URLs
        output_dir: Output directory
        languages: Preferred languages for transcripts
        max_concurrent: Maximum concurrent video processing

    Returns:
        List of results
    """
    print("🚀 Starting video transcript extraction...")
    print(f"   Videos to process: {len(video_urls)}")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"   Output directory: {output_dir}\n")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(video_url):
        async with semaphore:
            result = await video_to_markdown(video_url, output_dir, languages)
            # Small delay between videos
            await asyncio.sleep(0.5)
            return result

    # Process all videos concurrently
    tasks = [process_with_semaphore(url) for url in video_urls]
    results = await asyncio.gather(*tasks)

    return results
