import tensorflow as tf
import numpy as np
import librosa

def prepare_audio(audio_path, sr=22050, duration=5):
    """
    Prepare audio file for model prediction by converting it to mel spectrogram
    """
    # Load audio file and fix length
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
    audio = librosa.util.fix_length(audio, size=duration * sr)
    
    # Create mel spectrogram with same parameters as in training
    mel_spect = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    target_length = 109
    if mel_spect.shape[1] > target_length:
        mel_spect = mel_spect[:, :target_length]
    elif mel_spect.shape[1] < target_length:
        pad_width = target_length - mel_spect.shape[1]
        mel_spect = np.pad(mel_spect, ((0, 0), (0, pad_width)), mode='constant')
    
    # Convert to log scale
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Normalize
    mel_spect_norm = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())
    
    # Reshape for model input (add batch and channel dimensions)
    return np.expand_dims(mel_spect_norm, axis=[0, -1])

def predict_audio(model_path, audio_path):
    """
    Load model and make prediction on audio file
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare audio
    processed_audio = prepare_audio(audio_path)
    
    # Make prediction
    prediction = model.predict(processed_audio)
    
    return prediction[0]

if __name__ == "__main__":
    MODEL_PATH = "audio_classifier.h5"
    AUDIO_PATH = "assets/sounds/inspelning.wav"  
    
    # Get prediction
    result = predict_audio(MODEL_PATH, AUDIO_PATH)
    print("Prediction result:", result)  # To see what result contains
    
    # Assuming binary classification (real vs fake)
    if result[0] > 0.5:
        print(f"Audio is predicted to be FAKE with confidence: {result[0]:.2%}")
    else:
        print(f"Audio is predicted to be REAL with confidence: {(1-result[0]):.2%}")
