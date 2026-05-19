#!/usr/bin/env python3
"""
Sifmography Infinite Transcriber — Locally Powered by MLX & Whisper
"""

import os
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
import sys
import time
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# Ensure correct virtual environment packages are loaded
sys.path.insert(0, str(Path(__file__).parent / "Voice-Clone-Studio/venv/lib/python3.12/site-packages"))

import gradio as gr
import mlx.core as mx

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp"}

def extract_audio_from_video(video_path: str, progress=gr.Progress()) -> str:
    """Extract audio from video file using ffmpeg and resample to 16kHz mono WAV."""
    progress(0.1, desc="Analyzing video file...")
    temp_dir = Path(tempfile.gettempdir())
    output_path = str(temp_dir / f"extracted_{Path(video_path).stem}_{int(time.time())}.wav")
    
    # Run FFmpeg to extract audio
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                 # Disable video recording
        "-acodec", "pcm_s16le", # Output uncompressed 16-bit PCM WAV
        "-ar", "16000",        # Resample to 16000Hz (Whisper's native rate)
        "-ac", "1",            # Convert to Mono
        output_path
    ]
    
    progress(0.4, desc="Extracting audio stream...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        progress(0.9, desc="Audio extraction complete!")
        return output_path
    except subprocess.CalledProcessError as e:
        progress(1.0, desc="Failed to extract audio.")
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{e.stderr}")

def normalize_timestamp(t_str: str) -> str:
    """Normalize timestamp string into standard HH:MM:SS or SS format for yt-dlp."""
    t_str = t_str.strip()
    if not t_str:
        return ""
    if t_str.replace(".", "", 1).isdigit():
        return t_str
    
    parts = t_str.split(":")
    try:
        if len(parts) == 2:
            m = int(parts[0])
            s = int(parts[1])
            return f"00:{m:02d}:{s:02d}"
        elif len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2])
            return f"{h:02d}:{m:02d}:{s:02d}"
    except ValueError:
        pass
    return t_str

def download_audio_from_url(url: str, start_time: str = "", end_time: str = "", progress=gr.Progress()) -> str:
    """Download audio from a URL using yt-dlp, with optional section clipping, and convert/resample to 16kHz mono WAV."""
    progress(0.1, desc="Analyzing URL...")
    temp_dir = Path(tempfile.gettempdir())
    timestamp = int(time.time())
    output_template = str(temp_dir / f"yt_download_{timestamp}_%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", output_template
    ]
    
    start_time = start_time.strip() if start_time else ""
    end_time = end_time.strip() if end_time else ""
    
    if start_time or end_time:
        start_norm = normalize_timestamp(start_time) if start_time else "0"
        end_norm = normalize_timestamp(end_time) if end_time else "inf"
        section_arg = f"*{start_norm}-{end_norm}"
        cmd.extend(["--download-sections", section_arg])
        progress(0.2, desc=f"Configuring time range clip: {start_norm} to {end_norm}...")
        
    cmd.append(url)
    
    progress(0.3, desc="Downloading media range from link...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        downloaded_files = list(temp_dir.glob(f"yt_download_{timestamp}_*"))
        if not downloaded_files:
            raise FileNotFoundError("Could not locate the downloaded audio file in temp directory.")
        wav_path = str(downloaded_files[0])
        progress(0.9, desc="Download & conversion complete!")
        return wav_path
    except subprocess.CalledProcessError as e:
        progress(1.0, desc="Failed to download from link.")
        error_msg = e.stderr or e.stdout or "Unknown error"
        if len(error_msg) > 500:
            error_msg = error_msg[-500:]
        raise RuntimeError(f"Link download failed:\n{error_msg}")

def get_huggingface_cached_models():
    """List already downloaded whisper models in HuggingFace cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    models = []
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                name = item.name.replace("models--", "").replace("--", "/")
                if "whisper" in name.lower():
                    models.append(name)
    # Add default fallbacks if none detected
    if "mlx-community/whisper-tiny" not in models:
        models.append("mlx-community/whisper-tiny")
    if "mlx-community/whisper-large-v3-mlx" not in models:
        models.append("mlx-community/whisper-large-v3-mlx")
    return sorted(list(set(models)))

def format_time(seconds: float) -> str:
    """Format seconds into a beautiful HH:MM:SS or MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def unload_whisper_model():
    """Force garbage collection and clear MLX cache to free up VRAM/RAM immediately."""
    try:
        from mlx_whisper.transcribe import ModelHolder
        import gc
        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
        return "🧠 Model successfully unloaded from RAM/VRAM to conserve resources."
    except Exception as e:
        return f"⚠️ Failed to unload model: {e}"

def shutdown_app():
    """Gracefully shutdown the server after a short delay."""
    import threading
    def _shutdown():
        time.sleep(1.5)
        os._exit(0)
    threading.Thread(target=_shutdown).start()
    return "🛑 Server shutdown initiated! You can safely close this browser window.", "", "🔌 Server Offline"

def resolve_local_model_path(repo_id: str) -> str:
    """Resolves a Hugging Face repo ID to its absolute local snapshot path if cached, otherwise returns the repo ID."""
    try:
        from pathlib import Path
        folder_name = f"models--{repo_id.replace('/', '--')}"
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / folder_name / "snapshots"
        
        if cache_dir.exists():
            snapshots = sorted([d for d in cache_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
            if snapshots:
                return str(snapshots[0])
    except Exception:
        pass
    return repo_id

def process_transcription(
    file_path,
    url_input,
    clip_start,
    clip_end,
    model_name,
    custom_model,
    language_code,
    task,
    word_timestamps,
    work_offline,
    progress=gr.Progress()
):
    url_input = url_input.strip() if url_input else ""
    
    if not file_path and not url_input:
        return "⚠️ Please upload a file or enter an audio/video URL first.", "", "No input provided"

    # Set offline mode
    if work_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"

    # Determine final model to use
    raw_model = custom_model.strip() if custom_model.strip() else model_name
    selected_model = resolve_local_model_path(raw_model)
    
    # Import mlx_whisper locally to respect HF_HUB_OFFLINE state changes
    try:
        import mlx_whisper
    except ImportError:
        return (
            "❌ mlx_whisper could not be imported. Please make sure the venv is correctly configured.",
            "",
            "Import Error"
        )

    start_time = time.time()
    temp_wav_path = None
    status_logs = []
    
    try:
        if url_input:
            status_logs.append(f"🔗 Link detected: {url_input}")
            if clip_start or clip_end:
                status_logs.append(f"✂️ Requested clip range: {clip_start or 'Start'} to {clip_end or 'End'}")
            if work_offline:
                status_logs.append("⚠️ Note: Working Offline is enabled. If this link requires internet access, download may fail.")
            progress(0.05, desc="Initializing link download...")
            downloaded_wav = download_audio_from_url(
                url_input,
                start_time=clip_start,
                end_time=clip_end,
                progress=progress
            )
            temp_wav_path = downloaded_wav
            audio_target = downloaded_wav
            status_logs.append("📥 Successfully downloaded and converted link segment to audio stream.")
        else:
            original_path = Path(file_path)
            file_extension = original_path.suffix.lower()
            
            # Check if it's a video and extract audio
            if file_extension in VIDEO_EXTENSIONS:
                status_logs.append("🎥 Detected Video file. Extracting high-quality audio stream...")
                progress(0.1, desc="Extracting audio from video...")
                temp_wav_path = extract_audio_from_video(file_path, progress=progress)
                audio_target = temp_wav_path
                status_logs.append("🔊 Audio successfully extracted and resampled to 16kHz mono WAV.")
            else:
                status_logs.append("🎵 Audio file detected directly.")
                audio_target = file_path

        # Setup language parameters
        lang = "None"
        if language_code and language_code != "Auto Detect":
            if "(" in language_code:
                lang = language_code.split("(")[-1].replace(")", "").strip()
            else:
                lang = language_code.lower()
            status_logs.append(f"🌐 Forcing language: {language_code} ({lang})")
        else:
            status_logs.append("🌐 Automatically detecting language from first 30 seconds of audio...")

        if task == "Translate to English":
            status_logs.append("🔄 Task set to Translation (translating to English).")
        else:
            status_logs.append("📝 Task set to Transcription.")

        status_logs.append(f"🧠 Dispatching MLX Whisper subprocess: {selected_model}...")
        progress(0.4, desc="Preparing subprocess...")

        # Setup temporary result path
        temp_json_path = str(Path(tempfile.gettempdir()) / f"whisper_res_{int(time.time())}.json")
        
        # Build the inline Python code to run in the subprocess
        script_code = (
            "import os, sys, json, time;\n"
            "try:\n"
            "    # Add the current venv site-packages to sys.path in the subprocess\n"
            "    import pathlib\n"
            "    sys.path.insert(0, str(pathlib.Path(sys.argv[8]) / 'Voice-Clone-Studio/venv/lib/python3.12/site-packages'))\n"
            "    import mlx_whisper, mlx.core as mx;\n"
            "    audio_path = sys.argv[1]; model_name = sys.argv[2]; language = sys.argv[3];\n"
            "    task = sys.argv[4]; word_timestamps = sys.argv[5] == 'True';\n"
            "    work_offline = sys.argv[6] == 'True'; output_json_path = sys.argv[7];\n"
            "    if work_offline: os.environ['HF_HUB_OFFLINE'] = '1'\n"
            "    else: os.environ['HF_HUB_OFFLINE'] = '0'\n"
            "    kwargs = {'path_or_hf_repo': model_name, 'word_timestamps': word_timestamps}\n"
            "    if language and language != 'None': kwargs['language'] = language\n"
            "    if task == 'Translate to English': kwargs['task'] = 'translate'\n"
            "    else: kwargs['task'] = 'transcribe'\n"
            "    r = mlx_whisper.transcribe(audio_path, **kwargs)\n"
            "    mx.clear_cache()\n"
            "    with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(r, f, ensure_ascii=False)\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    import traceback; print(traceback.format_exc(), file=sys.stderr); sys.exit(1)\n"
        )
        
        # Execute the python script in a completely fresh process to avoid MPS thread locks
        python_exec = sys.executable
        cmd = [
            python_exec,
            "-c",
            script_code,
            audio_target,
            selected_model,
            lang,
            task,
            str(word_timestamps),
            str(work_offline),
            temp_json_path,
            str(Path(__file__).parent)
        ]
        
        progress(0.5, desc="Transcribing (running local MLX GPU engine)...")
        p_res = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check execution status
        if p_res.returncode != 0:
            raise RuntimeError(f"Whisper subprocess failed:\n{p_res.stderr}")
            
        progress(0.85, desc="Parsing transcription data...")
        
        # Load output
        if not os.path.exists(temp_json_path):
            raise RuntimeError("Result JSON file was not generated by subprocess.")
            
        with open(temp_json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            
        # Clean up temp file
        try:
            os.unlink(temp_json_path)
        except Exception:
            pass
            
        elapsed_time = time.time() - start_time
        status_logs.append(f"✅ Completed successfully in {elapsed_time:.2f} seconds!")
        status_logs.append("🧠 Subprocess completed and exited. MLX model is 100% unloaded and memory is perfectly clean.")
        
        # Format the main result text
        transcription_text = result.get("text", "").strip()
        
        # Parse detailed segments and word timestamps if selected
        detailed_output = []
        if word_timestamps:
            detailed_output.append("=== Word-Level Timestamps ===")
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    start = word.get("start", 0.0)
                    end = word.get("end", 0.0)
                    word_text = word.get("word", "").strip()
                    detailed_output.append(f"[{format_time(start)} -> {format_time(end)}] {word_text}")
        else:
            detailed_output.append("=== Segment-Level Timestamps ===")
            for segment in result.get("segments", []):
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                seg_text = segment.get("text", "").strip()
                detailed_output.append(f"[{format_time(start)} -> {format_time(end)}] {seg_text}")
                
        detailed_text = "\n".join(detailed_output)
        
        return (
            transcription_text,
            detailed_text,
            "\n".join(status_logs)
        )
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        status_logs.append(f"❌ Error occurred after {elapsed_time:.2f}s: {str(e)}")
        # Make sure we unload the model even on failure!
        unload_msg = unload_whisper_model()
        status_logs.append(unload_msg)
        return (
            f"❌ Transcription Failed.\n\nError details: {str(e)}",
            "",
            "\n".join(status_logs)
        )
    finally:
        # Clean up temp WAV files
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except Exception:
                pass

def export_transcript(text, format_type):
    """Generates a temporary file download for the transcription."""
    if not text or text.startswith("⚠️") or text.startswith("❌"):
        return None
        
    temp_dir = Path(tempfile.gettempdir())
    
    if format_type == "TXT":
        file_path = temp_dir / "transcription.txt"
        file_path.write_text(text, encoding="utf-8")
    elif format_type == "JSON":
        file_path = temp_dir / "transcription.json"
        data = {"transcription": text}
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
    return str(file_path)

# Custom premium styling
custom_css = """
body {
    background-color: #0d0e12;
    color: #f1f3f9;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 40px auto !important;
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.header-container {
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(135deg, rgba(29, 31, 43, 0.7) 0%, rgba(18, 20, 29, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #b993ff, #8a2be2, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.header-desc {
    color: #a0aec0;
    font-size: 1.1rem;
}
.glass-panel {
    background: rgba(18, 20, 29, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    padding: 20px;
    backdrop-filter: blur(4px);
}
.transcribe-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
.transcribe-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
}
.action-btn {
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
.action-btn:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    color: white !important;
}
"""

# Build premium Gradio App
with gr.Blocks(title="Sifmography Infinite Transcriber") as demo:
    # Beautiful Header
    with gr.Column(elem_classes=["header-container"]):
        gr.Markdown(
            "# ✨ Sifmography Infinite Transcriber",
            elem_classes=["header-title"]
        )
        gr.Markdown(
            "Locally-powered audio and video transcription built for Apple Silicon Macs. "
            "Simply upload files, record audio, or paste links (YouTube, Instagram Reels, etc.) to transcribe instantly.",
            elem_classes=["header-desc"]
        )
        
    with gr.Row():
        # Left Panel — Input and Configuration
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### 📂 Input & Config")
            
            # File Input supporting Audio and Video
            file_input = gr.File(
                label="Upload Audio or Video File",
                file_types=["audio", "video"],
                type="filepath",
                interactive=True
            )
            
            url_input = gr.Textbox(
                label="🔗 Or Enter Audio/Video Link (YouTube, Instagram Reels, SoundCloud, etc.)",
                placeholder="e.g. https://www.instagram.com/reel/CoY12345/ or YouTube URL",
                interactive=True
            )
            
            with gr.Row():
                clip_start_input = gr.Textbox(
                    label="✂️ Clip Start Time (Optional)",
                    placeholder="e.g. 00:01:30 or 90",
                    info="Clip start (HH:MM:SS or seconds)"
                )
                clip_end_input = gr.Textbox(
                    label="✂️ Clip End Time (Optional)",
                    placeholder="e.g. 00:05:00 or 300",
                    info="Clip end (HH:MM:SS or seconds)"
                )
            
            # Dynamic cached models dropdown
            cached_models = get_huggingface_cached_models()
            model_dropdown = gr.Dropdown(
                choices=cached_models,
                value=cached_models[0] if cached_models else "mlx-community/whisper-tiny",
                label="🧠 Choose local/cached Whisper Model",
                interactive=True,
                info="Models cached on your Mac are automatically detected."
            )
            
            custom_model_input = gr.Textbox(
                label="💡 Or input custom Hugging Face Model ID / Local Path",
                placeholder="e.g. mlx-community/whisper-base",
                interactive=True
            )
            
            # Languages Dropdown
            language_dropdown = gr.Dropdown(
                choices=[
                    "Auto Detect",
                    "English (en)", "Spanish (es)", "French (fr)", "German (de)",
                    "Japanese (ja)", "Chinese (zh)", "Korean (ko)", "Russian (ru)",
                    "Italian (it)", "Portuguese (pt)", "Dutch (nl)", "Turkish (tr)",
                    "Arabic (ar)", "Hindi (hi)", "Swedish (sv)", "Vietnamese (vi)"
                ],
                value="Auto Detect",
                label="🌐 Audio Language",
                interactive=True
            )
            
            # Task Selector
            task_dropdown = gr.Dropdown(
                choices=["Transcribe", "Translate to English"],
                value="Transcribe",
                label="🔄 Execution Task",
                interactive=True
            )
            
            # Advanced Configs
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                word_timestamps_cb = gr.Checkbox(
                    value=False,
                    label="📌 Output Word-level timestamps",
                    info="If checked, shows timestamps for individual words instead of sentences."
                )
                offline_mode_cb = gr.Checkbox(
                    value=True,
                    label="🔒 Work Offline",
                    info="Highly recommended. Prevents HuggingFace from making slow network checks that hang loading."
                )
                
            # Submit Button
            transcribe_btn = gr.Button("🚀 Start Transcription", elem_classes=["transcribe-btn"])
            
            # Shutdown Button
            shutdown_btn = gr.Button("🛑 Shutdown Server", variant="stop", elem_classes=["action-btn"])
            
        # Right Panel — Results
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### 📝 Results")
            
            with gr.Tabs():
                with gr.Tab("📄 Transcription"):
                    result_textbox = gr.Textbox(
                        label="Pure Text",
                        lines=16,
                        placeholder="Your transcribed text will appear here..."
                    )
                with gr.Tab("⏱️ Timestamps & Details"):
                    detailed_textbox = gr.Textbox(
                        label="Timed Transcript",
                        lines=16,
                        placeholder="Transcript with detailed segment or word timestamps..."
                    )
                with gr.Tab("📊 Processing Log"):
                    status_log = gr.Textbox(
                        label="Console Logs & Steps",
                        lines=8,
                        interactive=False,
                        placeholder="Log traces showing exact execution steps..."
                    )
            
            # Action Row
            with gr.Row():
                download_txt_btn = gr.Button("💾 Save as TXT", elem_classes=["action-btn"])
                download_json_btn = gr.Button("💾 Save as JSON", elem_classes=["action-btn"])
                
            # File download components (hidden until clicked)
            file_downloader = gr.File(label="Download File", visible=False)
            
            # Wire up download buttons
            def trigger_txt_download(text):
                path = export_transcript(text, "TXT")
                return gr.update(value=path, visible=True)
                
            def trigger_json_download(text):
                path = export_transcript(text, "JSON")
                return gr.update(value=path, visible=True)
 
            download_txt_btn.click(
                trigger_txt_download,
                inputs=[result_textbox],
                outputs=[file_downloader]
            )
            download_json_btn.click(
                trigger_json_download,
                inputs=[result_textbox],
                outputs=[file_downloader]
            )
 
    # Wire up Transcription action
    transcribe_btn.click(
        process_transcription,
        inputs=[
            file_input,
            url_input,
            clip_start_input,
            clip_end_input,
            model_dropdown,
            custom_model_input,
            language_dropdown,
            task_dropdown,
            word_timestamps_cb,
            offline_mode_cb
        ],
        outputs=[
            result_textbox,
            detailed_textbox,
            status_log
        ]
    )

    # Wire up Shutdown action
    shutdown_btn.click(
        shutdown_app,
        outputs=[
            result_textbox,
            detailed_textbox,
            status_log
        ]
    )
 
if __name__ == "__main__":
    # Start Gradio on a convenient port (8090)
    demo.launch(
        server_name="127.0.0.1",
        server_port=8090,
        quiet=True,
        css=custom_css
    )
