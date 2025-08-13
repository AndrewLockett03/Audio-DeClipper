import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file
file_path = librosa.util.example_audio_file()  # Use an example audio file
signal, sr = librosa.load(file_path)

# Compute the spectrogram
spectrogram = np.abs(librosa.stft(signal))
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
