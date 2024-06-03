import asyncio
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from tqdm import tqdm

async def get_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    video_id = re.search(r"(?<=v=)[^&]+", youtube_url)
    if not video_id:
        video_id = re.search(r"(?<=be/)[^&]+", youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    return video_id.group(0)

def get_transcript(video_id):
    """
    Retrieves the transcript of a YouTube video using its video ID.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
    except:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

    transcript = " ".join([entry['text'] for entry in transcript_list])
    return transcript

async def summarize_text(text, max_chunk_length=1024):
    """
    Summarizes the given text using a pre-trained model from Hugging Face.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in tqdm(text_chunks, desc="Summarizing"):
        summary = summarizer(chunk, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        await asyncio.sleep(0)  # Allow other tasks to run
    return " ".join(summaries)

async def main():
    youtube_url = input("Enter YouTube URL: ")

    print("Extracting video ID...")
    video_id = await get_video_id(youtube_url)

    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    
    print("\nTranscript of the video:\n")
    print(transcript + "\n")
    
    print("Summarizing the video...")
    summary = await summarize_text(transcript)

    print("\nSummary of the video:\n")
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())