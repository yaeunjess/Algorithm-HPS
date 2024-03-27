import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 오디오 파일 로드
y, sr = librosa.load('C_AcusticPlug26_1.wav')

# HPSS 적용
harmonic, percussive = librosa.effects.hpss(y)

# 조화음 출력 -> 진폭 출력
print(harmonic)
# 타격음 출력
print(percussive)

# 조화음과 타격음 부분 시각화
plt.figure(figsize=(12, 8))

D = librosa.stft(y)
D_harmonic, D_percussive = librosa.decompose.hpss(D)

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log')
plt.title('Full spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=np.max), y_axis='log')
plt.title('Harmonic spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=np.max), y_axis='log', x_axis='time')
plt.title('Percussive spectrogram')

plt.tight_layout()