import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

# This function analyzes the audio file and outputs the results to see if the audio has been manipulated
# It displays a picture with the waveform, spectrogram, RMS, MFCC, spectral centroid, transient strength and a graph with the spectral properties
def full_audio_analysis(file_path):
    print(f"Analysing file: {file_path}\n")

    # Load the audio file
    try:
        y, sr = librosa.load(file_path, sr=None)
        print(f"Sampling frequency: {sr} Hz")
        print(f"Length: {len(y) / sr:.2f} seconds")
        print(f"Number of samples: {len(y)}\n")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # RMS and ZCR are root mean square and zero crossing rate which shows how much the signal changes
    print("Basic analysis:")
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    print(f"RMS (Root Mean Square): {rms:.5f}")
    print(f"Zero Crossing Rate: {zcr:.5f}\n")

    # Spectral analysis is spectral centroid, bandwidth and flatness which means in laymens terms the average frequency, the bandwidth of the signal and how flat the signal is
    print("Spectral analysis:")
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr).mean()
    spectral_flatness = librosa.feature.spectral_flatness(S=stft).mean()
    print(f"Spectral centroid (average): {spectral_centroids.mean():.2f} Hz")
    print(f"Spektral bandwidth: {spectral_bandwidth:.2f} Hz")
    print(f"Spektral flatness: {spectral_flatness:.5f}\n")

    # MFCC analysis is short-term spectral analysis
    print("MFCC-analysis:")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"MFCC-coefficients (average): {np.mean(mfccs, axis=1)}\n")

    # Noise level (SNR) is the ratio between the signal strength and the noise strength
    print("Noise analysis (Signal-to-Noise Ratio):")
    noise_power = np.var(y[y < 0.01])  # Low amplitude as an approximation of noise
    signal_power = np.var(y)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    print(f"SNR: {snr:.2f} dB\n")

    # Harmonic analysis shows the energy in the harmonic and percussive parts of the signal
    print("Harmonic analysis:")
    harmonics, percussive = librosa.effects.hpss(y)
    print(f"Harmonic energy: {np.sum(harmonics**2):.5f}")
    print(f"Percussive energy: {np.sum(percussive**2):.5f}\n")

    # Transient analysis shows the strength of the signal at the beginning
    print("Transient analysis:")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    print(f"Number of transients (onsets): {len(onsets)}\n")

    # Plotting spectral properties
    print("Generating graphs...\n")
    plt.figure(figsize=(15, 10))

    # Waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    # Spectrogram shows a visual representation of the short-term spectral energy
    plt.subplot(3, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    # RMS is the square root of the mean of the squared values of the signal
    plt.subplot(3, 2, 3)
    rms = librosa.feature.rms(y=y)[0]
    times_rms = librosa.frames_to_time(range(len(rms)), sr=sr)
    plt.plot(times_rms, rms)
    plt.title("RMS (Energy)")
    plt.xlabel("Time (seconds)")

    # MFCC represents the short-term spectral energy
    plt.subplot(3, 2, 4)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title("MFCC")
    plt.colorbar()

    # Spectral Centroid is the average frequency of the signal
    plt.subplot(3, 2, 5)
    times_centroid = librosa.frames_to_time(range(len(spectral_centroids)), sr=sr)
    plt.semilogy(times_centroid, spectral_centroids, label='Spectral Centroid')
    plt.ylabel('Hz')
    plt.title('Spectral centroid')
    plt.legend()

    # Transient strength shows the strength of the signal transients
    plt.subplot(3, 2, 6)
    times_onset = librosa.frames_to_time(range(len(onset_env)), sr=sr)
    plt.plot(times_onset, onset_env, label='Onset Strength')
    plt.title('Transient strength')
    plt.xlabel('Time (seconds)')
    plt.legend()

    plt.tight_layout()
    plt.show()

file_path = r"C:\Users\nilss\Desktop\deepfake-detection\assets\sounds\inspelning.m4a"

if os.path.exists(file_path):
    full_audio_analysis(file_path)
else:
    print(f"File not found: {file_path}")
