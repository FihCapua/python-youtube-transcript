from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

def get_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    import re
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

def summarize_text(text):
    """
    Summarizes the given text using a pre-trained model from Hugging Face.
    """
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def main():
    youtube_url = input("Enter YouTube URL: ")
    video_id = get_video_id(youtube_url)
    transcript = get_transcript(video_id)
    
    print("\nTranscript of the video:\n")
    print(transcript + "\n")
    
    print("Summary of the video:")
    summary = summarize_text(transcript)
    print(summary)

if __name__ == "__main__":
    main()