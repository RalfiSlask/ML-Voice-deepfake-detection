import cv2
from moviepy.editor import VideoFileClip
from pymediainfo import MediaInfo
import subprocess
import os
import json
from datetime import timedelta

# Exiftool path - update this to your exiftool location
exiftool_path = r"C:\Users\nilss\Desktop\exiftool-13.03_64\exiftool.exe"

def format_timedelta(seconds):
    """Convert seconds to HH:MM:SS.ms format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds%60
    milliseconds = td.microseconds//1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def analyze_video_metadata(file_path, output_path):
    """Analyze video file metadata using multiple libraries"""
    metadata = {}
    
    # Basic file information
    metadata["file_info"] = {
        "filename": os.path.basename(file_path),
        "file_size": f"{os.path.getsize(file_path) / (1024*1024):.2f} MB",
        "file_path": file_path
    }

    # OpenCV Analysis
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            metadata["opencv"] = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
                "duration_seconds": float(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS))
            }
            metadata["opencv"]["duration_formatted"] = format_timedelta(metadata["opencv"]["duration_seconds"])
        cap.release()
    except Exception as e:
        metadata["opencv"] = f"Error: {str(e)}"

    # MoviePy Analysis
    try:
        video = VideoFileClip(file_path)
        metadata["moviepy"] = {
            "duration": video.duration,
            "duration_formatted": format_timedelta(video.duration),
            "size": video.size,
            "fps": video.fps,
            "has_audio": video.audio is not None,
            "rotation": video.rotation
        }
        video.close()
    except Exception as e:
        metadata["moviepy"] = f"Error: {str(e)}"

    # MediaInfo Analysis
    try:
        media_info = MediaInfo.parse(file_path)
        metadata["mediainfo"] = {
            "tracks": []
        }
        
        for track in media_info.tracks:
            track_data = track.to_data()
            metadata["mediainfo"]["tracks"].append({
                "track_type": track.track_type,
                "format": track_data.get("format"),
                "codec": track_data.get("codec"),
                "duration": track_data.get("duration"),
                "bit_rate": track_data.get("bit_rate"),
                "frame_rate": track_data.get("frame_rate"),
                "width": track_data.get("width"),
                "height": track_data.get("height"),
                "pixel_aspect_ratio": track_data.get("pixel_aspect_ratio"),
                "display_aspect_ratio": track_data.get("display_aspect_ratio")
            })
    except Exception as e:
        metadata["mediainfo"] = f"Error: {str(e)}"

    # ExifTool Analysis
    try:
        result = subprocess.run(
            [exiftool_path, "-j", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            metadata["exiftool"] = json.loads(result.stdout)[0]
        else:
            metadata["exiftool"] = f"Error: {result.stderr}"
    except Exception as e:
        metadata["exiftool"] = f"Error: {str(e)}"

    # Write results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Video Metadata Analysis Report\n\n")
        
        for analyzer, data in metadata.items():
            f.write(f"## {analyzer.upper()}\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        f.write(f"\n### {key}:\n")
                        f.write(f"```json\n{json.dumps(value, indent=2)}\n```\n")
                    else:
                        f.write(f"- **{key}:** {value}\n")
            else:
                f.write(f"{data}\n")
            f.write("\n")

    return metadata

if __name__ == "__main__":

    # Update these paths to match your project structure
    video_path = "assets/videos/video.mp4" 
    output_path = "video_metadata_report.md"
    
    if os.path.exists(video_path):
        print(f"Analyzing video: {video_path}")
        metadata = analyze_video_metadata(video_path, output_path)
        print(f"Analysis complete. Report saved to: {output_path}")
    else:
        print(f"Error: Video file not found at {video_path}") 