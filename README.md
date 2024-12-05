# Sound and Video Analysis Project

This project contains tools for analyzing audio and video files, including ML-based audio classification, spectrum analysis, and metadata extraction.
Primary focus is for deepfake detection and the focus is on the audio.

## Setup Instructions

### 1. Create and Activate Virtual Environment

#### Bash:

python -m venv sound_detection_env

## Activate an environment

sound_detection_env\Scripts\activate

## Install dependencies

Main dependencies:
- tensorflow==2.12.0
- numpy==1.23.5
- librosa==0.10.2.post1
- scikit-learn==1.3.2
- matplotlib==3.7.5
- opencv-python
- moviepy
- pymediainfo

See `requirements.txt` for complete list of dependencies.

pip install -r requirements.txt
pip list (for checking if all dependencies are installed)

Only the most neccecary dependencies are in the requirements.txt file, if one wants to add more, just add them to the file.
Might be conflicts with the versions, if so, try to find a compatible version or use different environments.

## Run script

python file_path.py

ex: python spectrum_analysis.py

## Project Structure

- `machine_learning.py` - ML model for audio classification
- `spectrum_analysis.py` - Audio spectrum analysis tools
- `sound_metadata.py` - Audio file metadata extraction
- `video_metadata.py` - Video file metadata extraction
- `test_ml_model.py` - Test script for the ML model

## Assets

Assets are in the `assets` folder and there we have sounds in the `sounds` folder and videos in the `videos` folder.

## Metadata Analysis

The metadata analysis is done by using different libraries and then comparing the results to see if the file has been manipulated.
The results are then written to a markdown file for easier reading.
Right now OpenAI is in the audio analysis, not currently used but is there if someone wants to use it, just remember to provide an API key for it.
We have metadata analysis both for audio and video.

## Machine Learning Model

Deepfake flac files are used from ASVspoof2021_DF sample, part00. 
link: https://www.asvspoof.org/index2021.html

Make sure to also download the keys and trial_metadata if not provided in the repo. Make a folder called `keys` and put the files in there.
If one wants to have a custom dataset, make sure all paths are updated in the `machine_learning.py` script.

## Models Already Trained

- `audio_classifier.h5` - Model that have been trained on large dataset of deepfake and real audio files
- `best_model.h5` - This model might be overfitted and not very good, so proceed with caution

## Notes
- Make sure to have Python 3.8 or later installed
- For video analysis, ensure you have ExifTool installed and update the path in `video_metadata.py`
- Audio files should be in WAV or MP3 format
- Video files should be in MP4 format

