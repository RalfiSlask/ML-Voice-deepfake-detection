import wave
import librosa
import soundfile as sf
from mutagen.mp3 import MP3
from pydub import AudioSegment
from pymediainfo import MediaInfo
import subprocess
import os


## In here we will analyze the metadata of the audio file to detect if the file has been manipulated

# This is done by analyzing the metadata of the file with different libraries and then comparing the results
# If the results are different, it is likely that the file has been manipulated
# This is done by comparing the metadata from different libraries and then drawing a conclusion
# We put everything in a markdown file to make it easier to read
# Exiftool is a command-line utility for reading and writing metadata to files
exiftool_path = r"C:\Users\nilss\Desktop\exiftool-13.03_64\exiftool.exe"

def full_metadata_analysis(file_path, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Analysing file: {file_path}\n\n")

        # Mutagen metadata
        file.write("Metadata from Mutagen:\n")
        try:
            audio = MP3(file_path)
            for attr in dir(audio.info):
                if not attr.startswith("_"):
                    file.write(f"{attr}: {getattr(audio.info, attr)}\n")
        except Exception as e:
            file.write(f"Mutagen analysis failed: {e}\n")
        file.write("\n")

        # Soundfile metadata
        file.write("Metadata from SoundFile (does not work for MP3):\n")
        try:
            with sf.SoundFile(file_path) as sf_file:
                file.write(f"Sampling frequency: {sf_file.samplerate}\n")
                file.write(f"Number of channels: {sf_file.channels}\n")
                file.write(f"Format: {sf_file.format}\n")
                file.write(f"Subformat: {sf_file.subtype}\n")
                file.write(f"Frames: {sf_file.frames}\n")
                file.write(f"Length: {sf_file.frames / sf_file.samplerate} seconds\n")
        except Exception as e:
            file.write(f"SoundFile analysis failed: {e}\n")
        file.write("\n")

        # Librosa metadata
        file.write("Metadata from Librosa:\n")
        try:
            y, sr = librosa.load(file_path, sr=None)
            file.write(f"Sampling frequency: {sr}\n")
            file.write(f"Number of samples: {len(y)}\n")
            file.write(f"Length: {len(y) / sr:.2f} seconds\n")
            file.write(f"RMS (Root Mean Square): {librosa.feature.rms(y=y).mean():.5f}\n")
            file.write(f"Tempo (BPM): {librosa.beat.tempo(y=y, sr=sr)[0]:.2f}\n")
        except Exception as e:
            file.write(f"Librosa analysis failed: {e}\n")
        file.write("\n")

        # Pydub metadata
        file.write("Metadata from Pydub:\n")
        try:
            audio = AudioSegment.from_file(file_path)
            file.write(f"Length: {len(audio) / 1000:.2f} seconds\n")
            file.write(f"Number of channels: {audio.channels}\n")
            file.write(f"Sampling frequency: {audio.frame_rate} Hz\n")
            file.write(f"Bits per sample: {audio.sample_width * 8}\n")
        except Exception as e:
            file.write(f"Pydub analysis failed: {e}\n")
        file.write("\n")

        # MediaInfo metadata
        file.write("Metadata from MediaInfo:\n")
        try:
            media_info = MediaInfo.parse(file_path)
            for track in media_info.tracks:
                file.write(f"Track type: {track.track_type}\n")
                for key, value in track.to_data().items():
                    file.write(f"  {key}: {value}\n")
        except Exception as e:
            file.write(f"MediaInfo analysis failed: {e}\n")
        file.write("\n")

        # ExifTool metadata
        file.write("Metadata from ExifTool:\n")
        try:
            result = subprocess.run(
                [exiftool_path, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                file.write(result.stdout)
            else:
                file.write(f"Error during ExifTool analysis: {result.stderr}\n")
        except Exception as e:
            file.write(f"ExifTool analysis failed: {e}\n")
        file.write("\n")

file_path_mp3 = r"C:\Users\nilss\Desktop\deepfake-detection\assets\sounds\inspelning.wav"
output_file_path = r"C:\Users\nilss\Desktop\deepfake-detection\metadata_output.md"

# Does the file exist?
if os.path.exists(file_path_mp3):
    full_metadata_analysis(file_path_mp3, output_file_path)
    print(f"Metadata has been saved to: {output_file_path}")
else:
    print(f"File not found: {file_path_mp3}")
