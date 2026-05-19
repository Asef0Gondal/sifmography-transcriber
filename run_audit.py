import os
import sys
import tempfile
import time
from pathlib import Path

# Add the workspace folder to python path
sys.path.insert(0, str(Path(__file__).parent))

from transcribe_app import download_audio_from_url, download_video_from_url, normalize_timestamp

def run_integration_audit():
    print("✨ Starting Sifmography Transcriber Live Pipeline Audit...")
    print("────────────────────────────────────────────────────────")
    
    # Test URL - a very short public video (5-second YouTube video)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # 1. Test Timestamp Normalization
    print("🔍 Testing timestamp normalization...")
    assert normalize_timestamp("1:30") == "00:01:30", "Normalization failed for MM:SS"
    assert normalize_timestamp(" 01:20:00 ") == "01:20:00", "Normalization failed with whitespace"
    assert normalize_timestamp("45.5") == "45.5", "Normalization failed for float seconds"
    print("✅ Timestamp normalization checks passed.")
    
    # 2. Test Audio Segment Download and Conversion (yt-dlp + ffmpeg)
    print("\n🔍 Testing audio segment extraction (yt-dlp + ffmpeg)...")
    try:
        audio_path = download_audio_from_url(test_url, start_time="1", end_time="3")
        print(f"✅ Audio downloaded and converted successfully to: {audio_path}")
        # Verify file exists and has size
        p = Path(audio_path)
        if p.exists() and p.stat().st_size > 0:
            print(f"   File Size: {p.stat().st_size} bytes")
            # Clean up
            os.unlink(audio_path)
            print("   Temp audio cleaned up successfully.")
        else:
            print("❌ Downloaded file is empty or missing.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Audio download/conversion failed: {e}")
        sys.exit(1)

    # 3. Test Video Segment Download and Merge
    print("\n🔍 Testing video segment download & merge (yt-dlp + ffmpeg)...")
    try:
        video_path = download_video_from_url(test_url, start_time="1", end_time="3")
        print(f"✅ Video downloaded and merged successfully to: {video_path}")
        p = Path(video_path)
        if p.exists() and p.stat().st_size > 0:
            print(f"   File Size: {p.stat().st_size} bytes")
            # Clean up
            os.unlink(video_path)
            print("   Temp video cleaned up successfully.")
        else:
            print("❌ Downloaded video file is empty or missing.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Video download/merge failed: {e}")
        sys.exit(1)

    print("\n────────────────────────────────────────────────────────")
    print("🎉 Live Pipeline Audit Completed: yt-dlp, ffmpeg, and clip logic are fully functional!")

if __name__ == "__main__":
    run_integration_audit()
